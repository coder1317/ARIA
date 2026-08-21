"""Knowledge Graph Memory — entities, relationships, and facts.

Upgrades ARIA's flat key-value memory to a graph structure where:
- Entities have types (person, project, tool, concept, etc.)
- Entities have attributes (key-value pairs)
- Relationships connect entities (works_on, uses, depends_on, etc.)
- Confidence scores track how sure ARIA is about each fact
- Temporal tracking knows when facts were learned and last verified

This is the foundation for ARIA to "understand" rather than just "remember".
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Entity:
    """A node in the knowledge graph."""
    id: int | None = None
    name: str = ""
    entity_type: str = "concept"  # person, project, tool, concept, location, etc.
    attributes: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.8  # 0-1, how sure we are
    source: str = ""  # where this knowledge came from
    created_at: str = ""
    updated_at: str = ""
    last_accessed: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "type": self.entity_type,
            "attributes": self.attributes, "confidence": self.confidence,
            "source": self.source, "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class Relationship:
    """An edge in the knowledge graph."""
    id: int | None = None
    source_id: int = 0
    target_id: int = 0
    relation_type: str = "related_to"  # works_on, uses, depends_on, etc.
    attributes: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.8
    source: str = ""
    created_at: str = ""


@dataclass
class GraphFact:
    """A fact extracted from the graph (entity + relationship + entity)."""
    subject: str
    subject_type: str
    relation: str
    object: str
    object_type: str
    confidence: float
    attributes: dict[str, str] = field(default_factory=dict)


class KnowledgeGraph:
    """Entity-relationship graph stored in SQLite.

    Usage:
        graph = KnowledgeGraph(db_path)

        # Add entities
        hari = graph.add_entity("Hari", "person", {"os": "Ubuntu", "language": "Python"})
        aria = graph.add_entity("ARIA", "project", {"lang": "Python", "status": "active"})

        # Add relationship
        graph.add_relationship(hari.id, aria.id, "builds", {"role": "creator"})

        # Query
        graph.find_entity("Hari")  # → Entity
        graph.find_related("Hari")  # → [(relationship, entity), ...]
        graph.get_facts_for("Hari")  # → [GraphFact, ...]
        graph.search("Python")  # → [Entity, ...]
    """

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self.db_path), timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS kg_entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL DEFAULT 'concept',
                attributes TEXT DEFAULT '{}',
                confidence REAL DEFAULT 0.8,
                source TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                last_accessed TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS kg_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                relation_type TEXT NOT NULL DEFAULT 'related_to',
                attributes TEXT DEFAULT '{}',
                confidence REAL DEFAULT 0.8,
                source TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (source_id) REFERENCES kg_entities(id),
                FOREIGN KEY (target_id) REFERENCES kg_entities(id)
            );

            CREATE INDEX IF NOT EXISTS idx_entity_name ON kg_entities(name);
            CREATE INDEX IF NOT EXISTS idx_entity_type ON kg_entities(entity_type);
            CREATE INDEX IF NOT EXISTS idx_rel_source ON kg_relationships(source_id);
            CREATE INDEX IF NOT EXISTS idx_rel_target ON kg_relationships(target_id);
            CREATE INDEX IF NOT EXISTS idx_rel_type ON kg_relationships(relation_type);
        """)
        conn.commit()

    # ── Entity CRUD ───────────────────────────────────────────

    def add_entity(self, name: str, entity_type: str = "concept",
                   attributes: dict[str, str] | None = None,
                   confidence: float = 0.8, source: str = "") -> Entity:
        """Add or update an entity. Returns existing if name+type matches."""
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        conn = self._get_conn()

        # Check if entity already exists
        row = conn.execute(
            "SELECT * FROM kg_entities WHERE name = ? AND entity_type = ?",
            (name, entity_type)
        ).fetchone()

        if row:
            # Update existing
            merged_attrs = json.loads(row["attributes"])
            if attributes:
                merged_attrs.update(attributes)
            conn.execute(
                "UPDATE kg_entities SET attributes = ?, confidence = ?, "
                "updated_at = ?, last_accessed = ? WHERE id = ?",
                (json.dumps(merged_attrs), max(confidence, row["confidence"]),
                 now, now, row["id"])
            )
            conn.commit()
            return Entity(
                id=row["id"], name=name, entity_type=entity_type,
                attributes=merged_attrs, confidence=confidence,
                source=source or row["source"],
                created_at=row["created_at"], updated_at=now,
            )

        # Insert new
        cur = conn.execute(
            "INSERT INTO kg_entities (name, entity_type, attributes, confidence, "
            "source, created_at, updated_at, last_accessed) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (name, entity_type, json.dumps(attributes or {}), confidence,
             source, now, now, now)
        )
        conn.commit()
        return Entity(
            id=cur.lastrowid, name=name, entity_type=entity_type,
            attributes=attributes or {}, confidence=confidence,
            source=source, created_at=now, updated_at=now,
        )

    def find_entity(self, name: str) -> Entity | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM kg_entities WHERE name = ? ORDER BY confidence DESC LIMIT 1",
            (name,)
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE kg_entities SET last_accessed = ? WHERE id = ?",
                (time.strftime("%Y-%m-%d %H:%M:%S"), row["id"])
            )
            conn.commit()
            return self._row_to_entity(row)
        return None

    def search(self, query: str, limit: int = 10) -> list[Entity]:
        """Search entities by name or attributes."""
        conn = self._get_conn()
        # Search by name (fuzzy)
        rows = conn.execute(
            "SELECT * FROM kg_entities WHERE name LIKE ? "
            "OR attributes LIKE ? ORDER BY confidence DESC LIMIT ?",
            (f"%{query}%", f"%{query}%", limit)
        ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    def list_by_type(self, entity_type: str, limit: int = 50) -> list[Entity]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM kg_entities WHERE entity_type = ? "
            "ORDER BY confidence DESC LIMIT ?",
            (entity_type, limit)
        ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    # ── Relationship CRUD ─────────────────────────────────────

    def add_relationship(self, source_id: int, target_id: int,
                        relation_type: str = "related_to",
                        attributes: dict[str, str] | None = None,
                        confidence: float = 0.8, source: str = "") -> Relationship:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        conn = self._get_conn()
        cur = conn.execute(
            "INSERT INTO kg_relationships (source_id, target_id, relation_type, "
            "attributes, confidence, source, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (source_id, target_id, relation_type,
             json.dumps(attributes or {}), confidence, source, now)
        )
        conn.commit()
        return Relationship(
            id=cur.lastrowid, source_id=source_id, target_id=target_id,
            relation_type=relation_type, attributes=attributes or {},
            confidence=confidence, source=source, created_at=now,
        )

    def find_related(self, entity_name: str) -> list[tuple[Relationship, Entity]]:
        """Find all entities related to the given entity."""
        entity = self.find_entity(entity_name)
        if not entity:
            return []
        conn = self._get_conn()
        # Outgoing relationships
        rows = conn.execute(
            """SELECT r.*, e.name, e.entity_type, e.attributes, e.confidence
               FROM kg_relationships r
               JOIN kg_entities e ON r.target_id = e.id
               WHERE r.source_id = ?
               ORDER BY r.confidence DESC""",
            (entity.id,)
        ).fetchall()
        # Incoming relationships
        rows += conn.execute(
            """SELECT r.*, e.name, e.entity_type, e.attributes, e.confidence
               FROM kg_relationships r
               JOIN kg_entities e ON r.source_id = e.id
               WHERE r.target_id = ?
               ORDER BY r.confidence DESC""",
            (entity.id,)
        ).fetchall()
        results = []
        for r in rows:
            rel = Relationship(
                id=r["id"], source_id=r["source_id"], target_id=r["target_id"],
                relation_type=r["relation_type"],
                attributes=json.loads(r["attributes"]),
                confidence=r["confidence"], source=r["source"],
                created_at=r["created_at"],
            )
            ent = Entity(
                name=r["name"], entity_type=r["entity_type"],
                attributes=json.loads(r["attributes"]),
                confidence=r["confidence"],
            )
            results.append((rel, ent))
        return results

    def get_facts_for(self, entity_name: str) -> list[GraphFact]:
        """Get all facts about an entity as readable triples."""
        related = self.find_related(entity_name)
        facts = []
        for rel, ent in related:
            facts.append(GraphFact(
                subject=entity_name,
                subject_type="",
                relation=rel.relation_type,
                object=ent.name,
                object_type=ent.entity_type,
                confidence=rel.confidence,
                attributes=rel.attributes,
            ))
        return facts

    # ── Graph queries ─────────────────────────────────────────

    def shortest_path(self, name_a: str, name_b: str, max_depth: int = 3) -> list[str] | None:
        """Find shortest path between two entities (BFS)."""
        a = self.find_entity(name_a)
        b = self.find_entity(name_b)
        if not a or not b:
            return None
        if a.id == b.id:
            return [name_a]

        conn = self._get_conn()
        visited = {a.id}
        queue = [(a.id, [name_a])]

        for _ in range(max_depth):
            next_queue = []
            for node_id, path in queue:
                # Get neighbors
                rows = conn.execute(
                    "SELECT target_id FROM kg_relationships WHERE source_id = ? "
                    "UNION "
                    "SELECT source_id FROM kg_relationships WHERE target_id = ?",
                    (node_id, node_id)
                ).fetchall()
                for (neighbor_id,) in rows:
                    if neighbor_id in visited:
                        continue
                    visited.add(neighbor_id)
                    neighbor = conn.execute(
                        "SELECT name FROM kg_entities WHERE id = ?",
                        (neighbor_id,)
                    ).fetchone()
                    if neighbor:
                        new_path = path + [neighbor["name"]]
                        if neighbor_id == b.id:
                            return new_path
                        next_queue.append((neighbor_id, new_path))
            queue = next_queue
        return None

    def subgraph(self, entity_name: str, depth: int = 1) -> dict:
        """Get a subgraph around an entity."""
        entity = self.find_entity(entity_name)
        if not entity:
            return {"entities": [], "relationships": []}

        entities = {entity.id: entity}
        related = self.find_related(entity_name)
        for rel, ent in related:
            if ent.name != entity_name:
                entities[ent.name] = ent

        return {
            "entities": [e.to_dict() for e in entities.values()],
            "relationships": [
                {"source": r[0].source_id, "target": r[0].target_id,
                 "type": r[0].relation_type, "confidence": r[0].confidence}
                for r in related
            ],
        }

    def stats(self) -> dict[str, int]:
        conn = self._get_conn()
        entities = conn.execute("SELECT COUNT(*) FROM kg_entities").fetchone()[0]
        relationships = conn.execute("SELECT COUNT(*) FROM kg_relationships").fetchone()[0]
        types = dict(conn.execute(
            "SELECT entity_type, COUNT(*) FROM kg_entities GROUP BY entity_type"
        ).fetchall())
        rel_types = dict(conn.execute(
            "SELECT relation_type, COUNT(*) FROM kg_relationships GROUP BY relation_type"
        ).fetchall())
        return {
            "entities": entities,
            "relationships": relationships,
            "entity_types": types,
            "relationship_types": rel_types,
        }

    # ── Auto-extract from text ────────────────────────────────

    def extract_and_store(self, text: str, source: str = "conversation") -> list[Entity]:
        """Simple extraction of entities from text (keyword-based).

        For production, this would use the LLM to extract structured entities.
        This basic version extracts capitalized words as potential entities.
        """
        import re
        # Find capitalized words (potential entity names)
        words = re.findall(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b', text)
        entities = []
        seen = set()
        for word in words:
            if word in seen or len(word) < 2:
                continue
            seen.add(word)
            # Simple type detection
            word_lower = word.lower()
            if word_lower in ("i", "we", "you", "they", "he", "she"):
                continue
            etype = "concept"
            if any(t in word_lower for t in ("mr", "mrs", "ms", "dr", "prof")):
                etype = "person"
            elif any(t in word_lower for t in ("inc", "llc", "corp", "ltd")):
                etype = "organization"
            elif any(t in word_lower for t in (".py", ".js", ".go", ".rs")):
                etype = "file"
            elif any(t in word_lower for t in ("http", "www", ".com", ".org")):
                etype = "url"

            ent = self.add_entity(word, etype, source=source, confidence=0.6)
            entities.append(ent)
        return entities

    def _row_to_entity(self, row: sqlite3.Row) -> Entity:
        return Entity(
            id=row["id"], name=row["name"], entity_type=row["entity_type"],
            attributes=json.loads(row["attributes"]),
            confidence=row["confidence"], source=row["source"],
            created_at=row["created_at"], updated_at=row["updated_at"],
            last_accessed=row["last_accessed"],
        )
