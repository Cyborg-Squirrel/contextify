#!/usr/bin/env python3
"""
file_traversal.py - fetches files patching a regex and reads file contents.
"""
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import PyPDF2
from docx import Document


@dataclass
class MatchedFile:
    """Dataclass representing a file from a specified directory traversal"""
    # The absolute file path
    absolute_path: Path
    # The file path relative to the search root
    relative_path: Path
    # SHA256 hash
    file_hash: str


class FileTraversal:
    """Class for performing file operations"""

    def traverse_file_tree(self, search_root: str, include_pattern: str) -> list[MatchedFile]:
        """
        Walk the directory tree rooted at *search_root* and return a list of all
        files which do not match the *include_pattern*.

        Parameters
        ----------
        search_root : str
            The root directory to start the walk from.
        include_pattern : str
            The regex of files to include.
        """
        root_path = Path(search_root)
        if not root_path.is_dir():
            print(f"'{search_root}' is not a directory or does not exist.")
            return []

        matched_files = []

        for root, _, files in os.walk(search_root):
            for file in files:
                full_file_path = Path(os.path.join(root, file))
                if not full_file_path.is_dir():
                    result = re.search(include_pattern, file)
                    if result is not None:
                        file_hash = self._get_file_hash(full_file_path)
                        if file_hash is not None:
                            relative_path = full_file_path.relative_to(root_path)
                            matched_files.append(MatchedFile(full_file_path,
                                                             relative_path, file_hash))

        return matched_files

    def _get_file_hash(self, path: Path, blocksize: int = 64 * 1024) -> Optional[str]:
        """
        Return the SHA-256 hash of the file at *path* as a 64-character hex string.
        """
        h = hashlib.sha256()
        try:
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(blocksize), b""):
                    h.update(chunk)
            return h.hexdigest()
        # pylint: disable=broad-exception-caught
        except Exception as ex:
            print(f"Encountered error {str(ex)} while making a hash of {str(path)}")
            return None

    def read_file_contents(self, path: Path) -> Optional[str]:
        """
        Reads the contents of the specified file, returns None if there
        is an error reading the file.
        """
        if path.is_dir():
            print(f"`{path}` is a directory, skipping.")
            return None
        absolute_path = str(path)

        try:
            if absolute_path.endswith(".pdf"):
                return self._extract_text_from_pdf(path)
            if absolute_path.endswith('.docx'):
                return self._extract_text_from_docx(path)
            with path.open("r", encoding="utf-8") as file:
                return file.read()
        # pylint: disable=broad-exception-caught
        except Exception as ex:
            print(f"Encountered error {str(ex)} while reading the contents of {str(path)}")
            return None

    def _extract_text_from_pdf(self, path: Path) -> str:
        """Extract text content from PDF files"""
        with path.open('rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()

    def _extract_text_from_docx(self, path: Path) -> str:
        """Extract text content from Word documents"""
        doc = Document(path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
