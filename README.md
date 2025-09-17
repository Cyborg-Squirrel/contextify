# Contextify

## Description

Contextify is a Python project that scans a single directory, a root directory and all subdirectories, or a single file. It supports creating image summaries using LLMs with vision capabilities, stores them, and allows user editing of these summaries before sending them to an embedding LLM. Embeddings are saved to Chroma and queried to support AI prompts. 
Requirements 

## Requirements

*   Python 3.10 or newer
  

## Installation

1.  Install dependencies:
    ```bash
    pip install ollama chromadb
    ```
   
2. Install Ollama and make sure the "expose Ollama to network" setting is enabled. It is disabled by default.

3.  Configure the project:
    *   Create a configuration file config.json. See config_template.json for a starting point.

## Usage

```bash
    python contextify.py
```

## Contributing

* Create a pull request
* Explain your changes
* The GitHub workflow must pass
