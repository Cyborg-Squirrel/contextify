#!/usr/bin/env python3
"""A script to embed files to enable RAG for AI prompts"""

import json

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

    def get_new_files(self, roots: list[str], context: str, include_pattern: str):
        """Gets all new or changed files in a root directory patching a specified pattern"""
        new_files = []
        for root in iter(roots):
            files = self.file_traversal.traverse_file_tree(root, include_pattern)
            for file in iter(files):
                relative_path = str(file.relative_path)
                chroma_file = self.chroma_api.get_file(relative_path, context)
                if chroma_file is not None:
                    if not chroma_file["metadatas"]["hash"] == file.file_hash:
                        files.append(file)
        return new_files

    def save_file(self, file: MatchedFile, context: str):
        """Generates and saves file embeddings in ChromaDb"""
        relative_path = file.relative_path
        file_contents = self.file_traversal.read_file_contents(file.absolute_path)
        if file_contents is not None:
            response = self.ollama_client.embed(
                 model=self.embedding_model,
                 input=file_contents,
                 )
            embeddings = response["embeddings"]
            self.chroma_api.add_file(context, str(relative_path), file.file_hash,
                                file_contents, 0, embeddings)

def main():
    """The main method"""
    config = {}
    with open('config.json', 'r', encoding='utf-8') as file:
        config = json.load(file)
    ollama_url = config["ollamaUrl"]
    contexts = config["contexts"]
    embedding_model = config["embeddingModel"]

    print(f"ollama url: {ollama_url}")
    print(f"embedding model: {embedding_model}")

    contextify = Contextify(ollama_url, embedding_model)

    for context in iter(contexts):
        roots = context["roots"]
        include_pattern = context["includePattern"]
        context_name = context["name"]

        print(f"Processing context: {context_name}")
        print(f"Roots: {roots}")
        print(f"Include pattern: {include_pattern}")

        new_files = contextify.get_new_files(roots, context_name, include_pattern)
        for new_file in iter(new_files):
            print(f"new file {new_file}")
            # contextify.save_file(new_file, context)

if __name__ == "__main__":
    main()
