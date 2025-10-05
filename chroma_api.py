#!/usr/bin/env python3
"""
chroma_api.py - reads and writes data to ChromaDB
"""

from typing import Optional

import chromadb
from chromadb.config import DEFAULT_DATABASE, DEFAULT_TENANT, Settings


class ChromaApi:
    """
    API class for interfacing with ChromaDB
    """
    def __init__(self):
        self.client = chromadb.Client(
            settings=Settings(),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE,
        )

    def _get_collection(self, context_name: str) -> chromadb.Collection:
        return self.client.get_or_create_collection(context_name)

    def get_file(self, relative_file_path: str, context_name: str) -> Optional[chromadb.GetResult]:
        """Gets the file from ChromaDB if it exists."""
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
                 contents: str, page_number: int, total_pages: int, embeddings: list[float]):
        """Saves a file to ChromaDB"""
        collection = self._get_collection(context_name)
        collection.add(
            embeddings=embeddings,
            documents=[contents],
            metadatas=[{
                "source": relative_file_path,
                "chunk_id": page_number,
                "total_chunks": total_pages,
                "hash": file_hash
            }],
            ids=[f"{relative_file_path}_{page_number}"]
        )

    def delete_file(self, context_name: str, relative_file_path: str):
        """Deletes a file in ChromaDB"""
        collection = self._get_collection(context_name)
        collection.delete(where={"$and": [{"source": {"$eq": relative_file_path}}]})

    def update_file(self, context_name: str, relative_file_path: str, file_hash: str,
                 contents: str, page_number: int, total_pages: int, embeddings: list[float]):
        """Updates a file in ChromaDB"""
        file = self.get_file(relative_file_path, context_name)
        if file is None:
            # pylint: disable=broad-exception-raised
            raise Exception(f"""Cannot update a file which has not been saved in the db
                            `{relative_file_path}`""")

        collection = self._get_collection(context_name)
        collection.update(
            embeddings=embeddings,
            documents=[contents],
            metadatas=[{
                "source": relative_file_path,
                "chunk_id": page_number,
                "total_chunks": total_pages,
                "hash": file_hash,
            }],
            ids=[f"{relative_file_path}_{page_number}"]
        )
