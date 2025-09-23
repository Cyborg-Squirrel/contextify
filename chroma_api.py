#!/usr/bin/env python3
"""
chroma_api.py - reads and writes data to ChromaDb
"""

from typing import Optional

import chromadb
from chromadb.config import DEFAULT_DATABASE, DEFAULT_TENANT, Settings


class ChromaApi:
    """
    API class for interfacing with ChromaDb
    """
    collection_base_name = "contextify"
    def __init__(self):
        self.client = chromadb.PersistentClient(
            settings=Settings(),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE,
        )

    def _get_collection(self, context_name: str) -> chromadb.Collection:
        return self.client.get_or_create_collection(
            f"{self.collection_base_name}-{context_name}"
        )

    def get_file(self, relative_file_path: str, context_name: str) -> Optional[chromadb.GetResult]:
        """Gets the file from ChromaDb if it exists."""
        collection = self._get_collection(context_name)
        result = collection.get(
            where={"source": {"$eq": relative_file_path}},
            include=["documents", "metadatas"]
        )
        if len(result["ids"]) > 0:
            return result
        return None

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def add_file(self, context_name: str, relative_file_path: str, file_hash: str,
                 contents: str, page_number: int, embeddings: list[float]):
        """Saves a file to ChromaDb"""
        collection = self._get_collection(context_name)
        collection.add(
            embeddings=embeddings,
            documents=[contents],
            metadatas=[{
                "source": relative_file_path,
                "chunk_id": page_number,
                "hash": file_hash,
            }],
            ids=[f"{relative_file_path}_{page_number}"]
        )
