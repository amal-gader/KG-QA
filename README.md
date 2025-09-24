# KG-QA



## Scholarly Question Answering over Structured and Unstructured Data

This repository contains the code for a system that enables question answering over heterogeneous scholarly data sources, 
combining structured knowledge graphs (KGs) and unstructured textual content. 
The system leverages state-of-the-art Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) 
techniques.

## Background

Knowledge Graphs are powerful tools for representing and storing structured information. They are widely used in applications such as recommender systems, semantic search, and question answering. However, KGs often lack rich textual content and are not designed to store or retrieve unstructured information.

In real-world scenarios, scholarly data exists in multiple forms:

Structured: metadata from sources like ORKG, DBLP, and other publication databases, providing information about authors, publications, topics, and citations.

Unstructured: full-text PDFs, scientific articles, and textual documents, which contain detailed content not captured in metadata.

KGQA systems have focused mainly on encyclopedic knowledge graphs such as DBpedia or Wikidata, leaving a gap in domains requiring reasoning over full-text scholarly content. Recent advances in LLMs and RAG methods provide opportunities to bridge this gap, enabling natural language queries over both structured and unstructured sources.

## System Overview

Our proposed system is an end-to-end pipeline for scholarly question answering:

1. Question Routing: Determine if a question should be answered using the Knowledge Graph or full-text content.

2. KG-based Questions:

- Convert natural language queries to SPARQL queries.

- Perform entity linking and template-based reasoning.

- Execute SPARQL queries against a local KG endpoint.

3. Content-based Questions:

- Use a RAG pipeline to retrieve relevant passages from full-text scholarly documents.

- Generate answers using LLMs.


## Benchmarks and Evaluation

Existing KGQA benchmarks focus primarily on general-purpose knowledge graphs. Our system is evaluated on real-world scholarly datasets, demonstrating its ability to:

- Bridge the gap between metadata and full-text content.

- Handle open-ended questions about publications and research topics.


## References
- DBLP: https://dblp.org
- ORKG: https://orkg.org
