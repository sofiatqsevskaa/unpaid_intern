from neo4j import GraphDatabase
from typing import List, Dict, Any
import uuid
import hashlib
import spacy
import os

from logger import get_logger

PASSWORD = os.environ.get("NEO4J_PASSWORD")
logger = get_logger("graph_storage")


class GraphStorage:
    def __init__(self, uri: str = "bolt://neo4j:7687",
                 username: str = "neo4j", password: str = PASSWORD):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        try:
            self.model = spacy.load("en_core_web_md")
            logger.info("spaCy model 'en_core_web_md' loaded successfully")
        except Exception as e:
            logger.warning(
                f"Could not load spaCy model, {e} entity extraction disabled")
            self.model = None

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        if self.model is None:
            return []
        doc = self.model(text)
        return [
            {"text": ent.text, "label": ent.label_,
                "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]

    def document_exists(self, user_id: str, content_hash: str) -> bool:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (u:User {id: $user_id})-[:UPLOADED]->(d:Document {content_hash: $content_hash})
                RETURN d.id as document_id, d.name as document_name
                LIMIT 1
                """,
                user_id=user_id,
                content_hash=content_hash
            )
            record = result.single()
            if record:
                logger.warning(
                    f"Duplicate detected in graph store, content_hash {content_hash[:12]}... "
                    f"already exists as '{record['document_name']}' (id: {record['document_id']})"
                )
                return True
        return False

    def add_document(self, user_id: str, document_name: str,
                     content: str, metadata: Dict[str, Any]):

        content_hash = hashlib.sha256(content.encode()).hexdigest()
        logger.info(
            f"Adding document to graph storage '{document_name}' for user {user_id}, hash: {content_hash[:12]}...")

        if self.document_exists(user_id, content_hash):
            logger.warning(
                f"Skipping '{document_name}' (duplicate)")
            return {"document_id": None, "entities_extracted": 0, "entities": [], "skipped": True, "reason": "duplicate"}

        document_id = str(uuid.uuid4())

        with self.driver.session() as session:
            result = session.run(
                """
                MERGE (u:User {id: $user_id})

                CREATE (d:Document {
                    id: $document_id,
                    name: $document_name,
                    content: $content,
                    content_hash: $content_hash,
                    upload_time: datetime(),
                    tags: $tags,
                    description: $description
                })

                MERGE (u)-[:UPLOADED]->(d)

                RETURN d.id as document_id
                """,
                user_id=user_id,
                document_id=document_id,
                document_name=document_name,
                content=content[:5000],
                content_hash=content_hash,
                tags=metadata.get("tags") or [],
                description=metadata.get("description") or ""
            )
            document_id = result.single()["document_id"]

        entities = self.extract_entities(content)
        logger.info(
            f"Extracted {len(entities)} entities from '{document_name}' using spaCy en_core_web_md")

        entity_count = 0
        with self.driver.session() as session:
            for entity in entities:
                session.run(
                    """
                    MERGE (e:Entity {name: $entity_name, type: $entity_type})
                    ON CREATE SET e.created_at = datetime()
                    WITH e
                    MATCH (d:Document {id: $document_id})
                    MERGE (d)-[:MENTIONS {context: $context, position: $position}]->(e)
                    """,
                    entity_name=entity["text"],
                    entity_type=entity["label"],
                    document_id=document_id,
                    context=content[max(0, entity["start"] - 50)
                                        :entity["end"] + 50],
                    position=entity["start"]
                )
                entity_count += 1

        logger.info(
            f"Stored '{document_name}' in graph with {entity_count} entity relationships")
        return {
            "document_id": document_id,
            "entities_extracted": entity_count,
            "entities": entities[:10],
            "skipped": False
        }

    def query(self, user_id: str, query_text: str):
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (u:User {id: $user_id})-[:UPLOADED]->(d:Document)
                WHERE d.content CONTAINS $query_text
                OR d.name CONTAINS $query_text
                OR ANY(tag IN d.tags WHERE tag CONTAINS $query_text)
                OPTIONAL MATCH (d)-[:MENTIONS]->(e:Entity)
                WITH d, COLLECT(DISTINCT e) as entities
                RETURN d, entities
                LIMIT 10
                """,
                user_id=user_id,
                query_text=query_text
            )
            results = []
            for record in result:
                doc = record["d"]
                entities = record["entities"]
                results.append({
                    "document": {
                        "id": doc["id"],
                        "name": doc["name"],
                        "content_preview": doc["content"][:200],
                        "upload_time": doc["upload_time"].isoformat()
                        if hasattr(doc["upload_time"], "isoformat")
                        else str(doc["upload_time"])
                    },
                    "entities": [
                        {"name": e["name"], "type": e["type"]}
                        for e in entities if e is not None
                    ]
                })
            return results

    def list_documents(self, user_id: str):
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (u:User {id: $user_id})-[:UPLOADED]->(d:Document)
                OPTIONAL MATCH (d)-[:MENTIONS]->(e:Entity)
                WITH d, COLLECT(DISTINCT e) as entities
                RETURN d, entities
                ORDER BY d.upload_time DESC
                LIMIT 50
                """,
                user_id=user_id
            )
            documents = []
            for record in result:
                doc = record["d"]
                documents.append({
                    "id": doc["id"],
                    "name": doc["name"],
                    "content_preview": doc["content"][:200],
                    "upload_time": doc["upload_time"].isoformat()
                    if hasattr(doc["upload_time"], "isoformat")
                    else str(doc["upload_time"])
                })
            return documents

    def close(self):
        self.driver.close()
