#!/usr/bin/env python3
"""A script to embed files to enable RAG for AI prompts"""

import argparse
import json
from typing import Optional

from ollama import Client

from chroma_api import ChromaApi
from file_traversal import FileTraversal, MatchedFile


class Contextify():
    """Class for coordinating high level logic for reading and embedding files"""
    file_traversal = FileTraversal()
    ollama_client: Client
    embedding_model: str
    chroma_api = ChromaApi()

    def __init__(self, ollama_url: str, embedding_model: str):
        self.ollama_client = Client(host=ollama_url)
        self.embedding_model = embedding_model

    def _traverse_file_trees(self, roots: list[str], include_pattern: str) -> list[MatchedFile]:
        """
        Traverses the specified root directories `roots`.
        
        Returns a list of files matching the `include_pattern` regex.
        """
        new_files = []
        for root in iter(roots):
            files = self.file_traversal.traverse_file_tree(root, include_pattern)
            new_files.extend(files)
        return new_files

    def get_new_or_changed_files(self, roots: list[str], context: str,
                                 include_pattern: str) -> dict[str, list[MatchedFile]]:
        """
        Gets all new or changed files in a root directory with filenames matching a
        specified `include_pattern` regex.
        """
        new_files = []
        updated_files = []
        files = self._traverse_file_trees(roots, include_pattern)
        for file in files:
            relative_path = str(file.relative_path)
            chroma_file = self.chroma_api.get_file(relative_path, context)
            if chroma_file is not None:
                metadatas = chroma_file.get("metadatas")
                metadata_hash = Optional[str]
                total_pages = Optional[str]
                if metadatas is not None:
                    for metadata in metadatas:
                        if "hash" in metadata.keys():
                            metadata_hash = metadata["hash"]
                        if "total_pages" in metadata.keys():
                            total_pages = metadata["total_pages"]
                        if metadata_hash is not None and total_pages is not None:
                            break
                    if metadata_hash != file.file_hash:
                        # print(f"{file.absolute_path} has been updated")
                        updated_files.append(file)
                    else:
                        # print(f"{file.absolute_path} is new")
                        new_files.append(file)
            else:
                # print(f"{file.absolute_path} is new")
                new_files.append(file)
        return {
            'new': new_files,
            'updated': updated_files
        }

    def _split_into_chunks(self, text: str, max_lines: int = 1000) -> list[str]:
        """Split `text` into a list of substrings, each containing at most `max_lines` lines."""
        lines = text.splitlines(True)
        chunks = []
        for i in range(0, len(lines), max_lines):
            chunk_lines = lines[i:i + max_lines]
            chunk_text = ''.join(chunk_lines)
            chunks.append(chunk_text)
        return chunks

    def delete_file(self, file: MatchedFile, context: str):
        """Deletes the file from ChromaDB matching `file` in context `context`."""
        self.chroma_api.delete_file(context, str(file.relative_path))

    def save_file(self, file: MatchedFile, context: str):
        """Generates and saves file embeddings in ChromaDb"""
        relative_path = file.relative_path
        file_contents = self.file_traversal.read_file_contents(file.absolute_path)
        if file_contents is not None:
            file_contents_as_list = self._split_into_chunks(file_contents)
            file_contents_as_list_len = len(file_contents_as_list)
            for i in range(file_contents_as_list_len):
                response = self.ollama_client.embed(
                    model=self.embedding_model,
                    input=file_contents,
                    )
                embeddings = response["embeddings"]
                self.chroma_api.add_file(context, str(relative_path), file.file_hash,
                                         file_contents, i, file_contents_as_list_len, embeddings)

    def query(self, query: str, context: str, n_results: int):
        """Queries the context `context` in ChromaDB database with `query`"""
        ollama_response = self.ollama_client.embed(
            model=self.embedding_model,
            input=query
        )
        print(ollama_response)
        return self.chroma_api.query(context, ollama_response['embeddings'], n_results=n_results)

# pylint: disable=too-many-locals
def main():
    """The main method"""
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument('-q', '--query', help='The query (prompt) to use')
    parser.add_argument('-c', '--context', help="""Specifies the context to use for a query""")
    parser.add_argument('-n', '--n-results', type=int, help="""
                        Specifies the max number of documents to return from the query""")
    parser.add_argument('-s', '--scan', action='store_true', help="""
                        Scans the configured directories and populates the ChromaDB collections""")
    args = parser.parse_args()

    config = {}
    with open('config.json', 'r', encoding='utf-8') as file:
        config = json.load(file)
    ollama_url = config["ollamaUrl"]
    contexts = config["contexts"]
    embedding_model = config["embeddingModel"]

    print(f"ollama url: {ollama_url}")
    print(f"embedding model: {embedding_model}")

    contextify = Contextify(ollama_url, embedding_model)

    if args.query is not None and args.context is not None:
        n_results = args.n_results if args.n_results is not None else 1
        print(f"Querying {args.context} with query {args.query} with n_results {n_results}")
        results = contextify.query(args.query, args.context, n_results)
        print(json.dumps(results))
    if args.scan:
        for context in iter(contexts):
            roots = context["roots"]
            include_pattern = context["includePattern"]
            context_name = context["name"]

            print(f"Processing context: {context_name}")
            print(f"Roots: {roots}")
            print(f"Include pattern: {include_pattern}")

            files_dict = contextify.get_new_or_changed_files(roots, context_name, include_pattern)
            new_files = files_dict['new']
            updated_files = files_dict['updated']

            print(f"New files {new_files}")
            print(f"Updated files {updated_files}")

            for new_file in iter(new_files):
                contextify.save_file(new_file, context_name)
            for updated_file in iter(updated_files):
                contextify.delete_file(updated_file, context_name)
                contextify.save_file(updated_file, context_name)

if __name__ == "__main__":
    main()
