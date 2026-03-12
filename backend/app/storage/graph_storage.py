from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional, Tuple
import uuid
import hashlib
import spacy
import os
import re

from logger import get_logger

PASSWORD = os.environ.get("NEO4J_PASSWORD")
logger = get_logger("graph_storage")

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


class GraphStorage:
    def __init__(
        self,
        uri: str = "bolt://neo4j:7687",
        username: str = "neo4j",
        password: str = PASSWORD,
    ):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

        try:
            self.model = spacy.load("en_core_web_md")
            logger.info("spaCy model 'en_core_web_md' loaded successfully")
        except Exception as e:
            logger.warning(
                f"Could not load spaCy model: {e} — entity extraction disabled")
            self.model = None

        self._init_schema()

    def _init_schema(self):
        constraints_and_indexes = [
            "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.type) IS NODE KEY",
            "CREATE INDEX doc_hash IF NOT EXISTS FOR (d:Document) ON (d.content_hash)",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX chunk_doc IF NOT EXISTS FOR (c:Chunk) ON (c.document_id)",
        ]
        with self.driver.session() as session:
            for stmt in constraints_and_indexes:
                try:
                    session.run(stmt)
                except Exception as e:
                    logger.warning(
                        f"Schema statement skipped (may already exist): {e}")
        logger.info("Neo4j schema initialized")

    def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        # splits to overlapping chunks, sentences if possible
        text = re.sub(r'\s+', ' ', text).strip()

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + CHUNK_SIZE

            if end < len(text):
                boundary = text.rfind('.', start + CHUNK_SIZE // 2, end)
                if boundary == -1:
                    boundary = text.rfind(' ', start + CHUNK_SIZE // 2, end)
                if boundary != -1:
                    end = boundary + 1

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "index": chunk_index,
                    "text": chunk_text,
                    "start_char": start,
                    "end_char": end,
                })
                chunk_index += 1

            start = end - CHUNK_OVERLAP

        return chunks

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        if self.model is None:
            return []
        doc = self.model(text)
        seen = set()
        entities = []
        for ent in doc.ents:
            key = (ent.text.strip().lower(), ent.label_)
            if key in seen:
                continue
            seen.add(key)
            entities.append({
                "text": ent.text.strip(),
                "normalized": ent.text.strip().lower(),
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })
        return entities

    def _extract_entity_relations(self, text: str) -> List[Tuple[str, str, str, str, str]]:
        if self.model is None:
            return []

        doc = self.model(text)
        relations = []

        for sent in doc.sents:
            sent_ents = [e for e in sent.ents]
            for i, e1 in enumerate(sent_ents):
                for e2 in sent_ents[i + 1:]:
                    if e1.text.strip().lower() != e2.text.strip().lower():
                        relations.append((
                            e1.text.strip(), e1.label_,
                            "CO_OCCURS_WITH",
                            e2.text.strip(), e2.label_,
                        ))

        return relations

    def _document_exists(self, user_id: str, content_hash: str) -> bool:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (u:User {id: $user_id})-[:UPLOADED]->(d:Document {content_hash: $content_hash})
                RETURN d.id as document_id, d.name as document_name
                LIMIT 1
                """,
                user_id=user_id,
                content_hash=content_hash,
            )
            record = result.single()
            if record:
                logger.warning(
                    f"Duplicate: hash {content_hash[:12]}... already exists "
                    f"as '{record['document_name']}' (id: {record['document_id']})"
                )
                return True
        return False

    def add_document(
        self,
        user_id: str,
        document_name: str,
        content: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:

        content_hash = hashlib.sha256(content.encode()).hexdigest()
        logger.info(
            f"adding '{document_name}' for user {user_id}, hash: {content_hash[:12]}")

        if self._document_exists(user_id, content_hash):
            return {
                "document_id": None,
                "chunks_stored": 0,
                "entities_extracted": 0,
                "skipped": True,
                "reason": "duplicate",
            }

        document_id = str(uuid.uuid4())

        with self.driver.session() as session:
            session.run(
                """
                MERGE (u:User {id: $user_id})
                CREATE (d:Document {
                    id: $document_id,
                    name: $document_name,
                    content_hash: $content_hash,
                    upload_time: datetime(),
                    tags: $tags,
                    description: $description,
                    char_count: $char_count
                })
                MERGE (u)-[:UPLOADED]->(d)
                """,
                user_id=user_id,
                document_id=document_id,
                document_name=document_name,
                content_hash=content_hash,
                tags=metadata.get("tags") or [],
                description=metadata.get("description") or "",
                char_count=len(content),
            )

        chunks = self._chunk_text(content)
        logger.info(f"Created {len(chunks)} chunks for '{document_name}'")

        with self.driver.session() as session:
            for chunk in chunks:
                chunk_id = str(uuid.uuid4())
                session.run(
                    """
                    MATCH (d:Document {id: $document_id})
                    CREATE (c:Chunk {
                        id: $chunk_id,
                        document_id: $document_id,
                        index: $chunk_index,
                        text: $text,
                        start_char: $start_char,
                        end_char: $end_char
                    })
                    CREATE (d)-[:HAS_CHUNK {index: $chunk_index}]->(c)
                    """,
                    document_id=document_id,
                    chunk_id=chunk_id,
                    chunk_index=chunk["index"],
                    text=chunk["text"],
                    start_char=chunk["start_char"],
                    end_char=chunk["end_char"],
                )

                if chunk["index"] > 0:
                    session.run(
                        """
                        MATCH (d:Document {id: $document_id})-[:HAS_CHUNK {index: $prev_index}]->(prev:Chunk)
                        MATCH (d)-[:HAS_CHUNK {index: $curr_index}]->(curr:Chunk)
                        CREATE (prev)-[:NEXT]->(curr)
                        """,
                        document_id=document_id,
                        prev_index=chunk["index"] - 1,
                        curr_index=chunk["index"],
                    )

        total_entities = 0
        with self.driver.session() as session:
            for chunk in chunks:
                entities = self._extract_entities(chunk["text"])
                for ent in entities:
                    session.run(
                        """
                        MERGE (e:Entity {name: $name, type: $type})
                        ON CREATE SET
                            e.normalized = $normalized,
                            e.created_at = datetime()
                        WITH e
                        MATCH (d:Document {id: $document_id})-[:HAS_CHUNK {index: $chunk_index}]->(c:Chunk)
                        MERGE (c)-[:MENTIONS {position: $position}]->(e)
                        MERGE (d)-[:MENTIONS]->(e)
                        """,
                        name=ent["text"],
                        type=ent["label"],
                        normalized=ent["normalized"],
                        document_id=document_id,
                        chunk_index=chunk["index"],
                        position=ent["start"],
                    )
                    total_entities += 1

                relations = self._extract_entity_relations(chunk["text"])
                for e1_text, e1_type, rel_type, e2_text, e2_type in relations:
                    session.run(
                        """
                        MERGE (e1:Entity {name: $e1_name, type: $e1_type})
                        MERGE (e2:Entity {name: $e2_name, type: $e2_type})
                        MERGE (e1)-[r:CO_OCCURS_WITH]->(e2)
                        ON CREATE SET r.count = 1, r.first_seen = datetime()
                        ON MATCH SET r.count = r.count + 1
                        """,
                        e1_name=e1_text,
                        e1_type=e1_type,
                        e2_name=e2_text,
                        e2_type=e2_type,
                    )

        logger.info(
            f"Stored '{document_name}': {len(chunks)} chunks, {total_entities} entity links"
        )
        return {
            "document_id": document_id,
            "chunks_stored": len(chunks),
            "entities_extracted": total_entities,
            "skipped": False,
        }

    def query(self, user_id: str, query_text: str) -> List[Dict[str, Any]]:
        """
        Query strategy:
        1. extract entities from the query
        2. if entities found, match chunks that mention those entities,
        3. expand with CO_OCCURS_WITH to find related entities and more chunks
        3. if no entities, fall back to text containment search on chunks
        returns chunks
        """
        query_entities = self._extract_entities(query_text)

        with self.driver.session() as session:
            if query_entities:
                entity_names = [e["text"] for e in query_entities]
                entity_normalized = [e["normalized"] for e in query_entities]

                result = session.run(
                    """
                    MATCH (u:User {id: $user_id})-[:UPLOADED]->(d:Document)-[:HAS_CHUNK]->(c:Chunk)
                    MATCH (c)-[:MENTIONS]->(e:Entity)
                    WHERE e.name IN $entity_names OR e.normalized IN $entity_normalized
                    WITH c, d, COLLECT(DISTINCT e) as direct_entities, COUNT(DISTINCT e) as direct_score

                    OPTIONAL MATCH (c)-[:MENTIONS]->(e2:Entity)-[:CO_OCCURS_WITH]-(related:Entity)
                    WITH c, d, direct_entities, direct_score,
                         COLLECT(DISTINCT related) as expanded_entities

                    RETURN c, d,
                           direct_entities,
                           expanded_entities,
                           direct_score
                    ORDER BY direct_score DESC
                    LIMIT 15
                    """,
                    user_id=user_id,
                    entity_names=entity_names,
                    entity_normalized=entity_normalized,
                )
            else:
                result = session.run(
                    """
                    MATCH (u:User {id: $user_id})-[:UPLOADED]->(d:Document)-[:HAS_CHUNK]->(c:Chunk)
                    WHERE toLower(c.text) CONTAINS toLower($query_text)
                    OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
                    WITH c, d, COLLECT(DISTINCT e) as direct_entities
                    RETURN c, d, direct_entities, [] as expanded_entities, 0 as direct_score
                    LIMIT 15
                    """,
                    user_id=user_id,
                    query_text=query_text,
                )

            results = []
            for record in result:
                chunk = record["c"]
                doc = record["d"]
                direct_ents = record["direct_entities"]
                expanded_ents = record["expanded_entities"]

                results.append({
                    "chunk": {
                        "id": chunk["id"],
                        "text": chunk["text"],
                        "index": chunk["index"],
                    },
                    "document": {
                        "id": doc["id"],
                        "name": doc["name"],
                        "upload_time": (
                            doc["upload_time"].isoformat()
                            if hasattr(doc["upload_time"], "isoformat")
                            else str(doc["upload_time"])
                        ),
                    },
                    "entities": {
                        "direct": [
                            {"name": e["name"], "type": e["type"]}
                            for e in direct_ents if e is not None
                        ],
                        "expanded": [
                            {"name": e["name"], "type": e["type"]}
                            for e in expanded_ents if e is not None
                        ],
                    },
                    "score": record["direct_score"],
                })

            return results

    # future work, when more information needed on subject

    def query_with_context(self, user_id: str, query_text: str) -> List[Dict[str, Any]]:
        results = self.query(user_id, query_text)

        enriched = []
        with self.driver.session() as session:
            for r in results:
                chunk_id = r["chunk"]["id"]

                context_result = session.run(
                    """
                    MATCH (c:Chunk {id: $chunk_id})
                    OPTIONAL MATCH (prev:Chunk)-[:NEXT]->(c)
                    OPTIONAL MATCH (c)-[:NEXT]->(next:Chunk)
                    RETURN prev.text as prev_text, next.text as next_text
                    """,
                    chunk_id=chunk_id,
                )
                ctx = context_result.single()
                r["context"] = {
                    "prev_chunk": ctx["prev_text"] if ctx else None,
                    "next_chunk": ctx["next_text"] if ctx else None,
                }
                enriched.append(r)

        return enriched

    def list_documents(self, user_id: str) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (u:User {id: $user_id})-[:UPLOADED]->(d:Document)
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                OPTIONAL MATCH (d)-[:MENTIONS]->(e:Entity)
                WITH d,
                     COUNT(DISTINCT c) as chunk_count,
                     COUNT(DISTINCT e) as entity_count
                RETURN d, chunk_count, entity_count
                ORDER BY d.upload_time DESC
                LIMIT 50
                """,
                user_id=user_id,
            )
            documents = []
            for record in result:
                doc = record["d"]
                documents.append({
                    "id": doc["id"],
                    "name": doc["name"],
                    "upload_time": (
                        doc["upload_time"].isoformat()
                        if hasattr(doc["upload_time"], "isoformat")
                        else str(doc["upload_time"])
                    ),
                    "tags": doc.get("tags", []),
                    "description": doc.get("description", ""),
                    "char_count": doc.get("char_count", 0),
                    "chunk_count": record["chunk_count"],
                    "entity_count": record["entity_count"],
                })
            return documents

    def get_entity_graph(self, user_id: str, entity_name: str, depth: int = 2) -> Dict[str, Any]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (u:User {id: $user_id})-[:UPLOADED]->(d:Document)-[:MENTIONS]->(root:Entity)
                WHERE toLower(root.name) CONTAINS toLower($entity_name)
                WITH root LIMIT 1
                CALL apoc.path.subgraphAll(root, {
                    relationshipFilter: 'CO_OCCURS_WITH',
                    maxLevel: $depth
                })
                YIELD nodes, relationships
                RETURN nodes, relationships
                """,
                user_id=user_id,
                entity_name=entity_name,
                depth=depth,
            )
            record = result.single()
            if not record:
                return {"nodes": [], "edges": []}

            nodes = [{"id": n["name"], "type": n["type"]}
                     for n in record["nodes"]]
            edges = [
                {
                    "from": r.start_node["name"],
                    "to": r.end_node["name"],
                    "count": r.get("count", 1),
                }
                for r in record["relationships"]
            ]
            return {"nodes": nodes, "edges": edges}

    def delete_document(self, user_id: str, document_id: str) -> bool:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (u:User {id: $user_id})-[:UPLOADED]->(d:Document {id: $document_id})
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                DETACH DELETE d, c
                RETURN COUNT(d) as deleted
                """,
                user_id=user_id,
                document_id=document_id,
            )
            record = result.single()
            deleted = record and record["deleted"] > 0
            if deleted:
                logger.info(f"Deleted document {document_id} and its chunks")
            return deleted

    def close(self):
        self.driver.close()
