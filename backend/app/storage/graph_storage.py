from neo4j import GraphDatabase
from typing import List, Dict, Any
import uuid
import spacy
import os

PASSWORD = os.environ.get("PASSWORD")


class GraphStorage:
    def __init__(self, uri: str = "bolt://localhost:7687",
                 username: str = "neo4j", password: str = PASSWORD):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        try:
            self.model = spacy.load("en_core_web_md")
        except:
            self.model = None

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        if self.model is None:
            return []

        doc = self.model(text)
        entities = []

        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })

        return entities

    def add_document(self, user_id: str, document_name: str,
                     content: str, metadata: Dict[str, Any]):

        document_id = str(uuid.uuid4())

        with self.driver.session() as session:

            result = session.run(
                """
                MERGE (u:User {id: $user_id})

                CREATE (d:Document {
                    id: $document_id,
                    name: $document_name,
                    content: $content,
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
                tags=metadata.get("tags", []),
                description=metadata.get("description", "")
            )

            document_id = result.single()["document_id"]

            entities = self.extract_entities(content)

            entity_count = 0

            for entity in entities:

                session.run(
                    """
                    MERGE (e:Entity {
                        name: $entity_name,
                        type: $entity_type
                    })

                    ON CREATE SET e.created_at = datetime()

                    WITH e

                    MATCH (d:Document {
                        id: $document_id
                    })

                    MERGE (d)-[:MENTIONS {
                        context: $context,
                        position: $position
                    }]->(e)
                    """,
                    entity_name=entity["text"],
                    entity_type=entity["label"],
                    document_id=document_id,
                    context=content[max(0, entity["start"] - 50):entity["end"] + 50],
                    position=entity["start"]
                )

                entity_count += 1

            return {
                "document_id": document_id,
                "entities_extracted": entity_count,
                "entities": entities[:10]
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

                RETURN d, COLLECT(DISTINCT e) as entities

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
                        {
                            "name": e["name"],
                            "type": e["type"]
                        }
                        for e in entities if e is not None
                    ]
                })

            return results

    def close(self):
        self.driver.close()
