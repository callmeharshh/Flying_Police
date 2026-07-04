from typing import List, Optional
import chromadb
from chromadb.utils import embedding_functions
from config import CHROMA_DIR, EMBEDDING_MODEL


class FrameIndex:
    def __init__(self, chroma_dir: str = None):
        self._client = chromadb.PersistentClient(path=chroma_dir or CHROMA_DIR)
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self._collection = self._client.get_or_create_collection(
            name="frames",
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def add_frame(
        self,
        frame_id: int,
        description: str,
        timestamp: str,
        location: str,
        objects: List[str],
        threat_level: str,
        bbox: Optional[tuple] = None,
        track_id: Optional[str] = None,
    ) -> None:
        metadata = {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "location": location,
            "objects": ",".join(objects),
            "threat_level": threat_level,
        }
        if bbox is not None:
            metadata["bbox_x"] = int(bbox[0])
            metadata["bbox_y"] = int(bbox[1])
            metadata["bbox_w"] = int(bbox[2])
            metadata["bbox_h"] = int(bbox[3])
            metadata["center_x"] = round(bbox[0] + bbox[2] / 2, 1)
            metadata["center_y"] = round(bbox[1] + bbox[3] / 2, 1)
        if track_id:
            metadata["track_id"] = track_id

        self._collection.upsert(
            ids=[str(frame_id)],
            documents=[description],
            metadatas=[metadata],
        )

    def query(self, text: str, n_results: int = 5,
              location: Optional[str] = None,
              threat_level: Optional[str] = None) -> List[dict]:
        where = {}
        if location:
            where["location"] = location
        if threat_level:
            where["threat_level"] = threat_level

        kwargs = dict(query_texts=[text], n_results=min(n_results, self._collection.count()))
        if where:
            kwargs["where"] = where

        if self._collection.count() == 0:
            return []

        results = self._collection.query(**kwargs)
        output = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            entry = {
                "frame_id": meta["frame_id"],
                "description": doc,
                "timestamp": meta["timestamp"],
                "location": meta["location"],
                "objects": meta["objects"],
                "threat_level": meta["threat_level"],
                "score": round(1 - results["distances"][0][i], 3),
            }
            if "center_x" in meta:
                entry["center_x"] = meta["center_x"]
                entry["center_y"] = meta["center_y"]
            if "track_id" in meta:
                entry["track_id"] = meta["track_id"]
            output.append(entry)
        return output

    def get_recent_at_location(self, location: str, n_results: int = 5) -> List[dict]:
        if self._collection.count() == 0:
            return []

        results = self._collection.get(
            where={"location": location},
            include=["metadatas", "documents"],
        )
        if not results["ids"]:
            return []

        entries = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            entry = {
                "frame_id": meta["frame_id"],
                "description": doc,
                "timestamp": meta["timestamp"],
                "location": meta["location"],
                "objects": meta.get("objects", ""),
            }
            if "center_x" in meta:
                entry["center_x"] = meta["center_x"]
                entry["center_y"] = meta["center_y"]
            if "track_id" in meta:
                entry["track_id"] = meta["track_id"]
            entries.append(entry)

        entries.sort(key=lambda e: e["frame_id"], reverse=True)
        return entries[:n_results]

    def list_all_frames(self) -> List[dict]:
        if self._collection.count() == 0:
            return []

        results = self._collection.get(include=["metadatas", "documents"])
        entries = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            entry = {
                "frame_id": meta["frame_id"],
                "description": doc,
                "timestamp": meta["timestamp"],
                "location": meta["location"],
                "objects": meta.get("objects", ""),
                "threat_level": meta["threat_level"],
            }
            if "center_x" in meta:
                entry["center_x"] = meta["center_x"]
                entry["center_y"] = meta["center_y"]
            if "track_id" in meta:
                entry["track_id"] = meta["track_id"]
            entries.append(entry)

        entries.sort(key=lambda e: e["frame_id"])
        return entries

    def count(self) -> int:
        return self._collection.count()
