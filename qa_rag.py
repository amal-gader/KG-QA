# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RAG (Retrieval Augmented Generation) Implementation with LlamaIndex
================================================================

This script demonstrates a RAG system using:
- LlamaIndex: For document indexing and retrieval
- Milvus: As vector store backend

Features:
1. Document Loading & Processing
2. Embedding & Storage
3. Query Processing

Notes:
    - First run may take time to download models
"""

import argparse
from argparse import Namespace
from datasets import load_dataset
import pandas as pd
from typing import Any

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.readers.web import SimpleWebPageReader
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike



import json
import re

from dotenv import load_dotenv
import os

from tqdm import tqdm

import nl2sparql
from utils import postprocess_sparql, title_to_filename, add_answer


load_dotenv()
api_key = os.getenv("UNI_API_KEY")

with open('DBLP-QuAD/DBLP-QuAD/similar_questions.json', 'r') as f:
    sim_questions = json.load(f)



def init_config(args: Namespace):
    """Initialize configuration with command line arguments"""
    return {
        "db_path": args.db_path,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "top_k": args.top_k,
    }


def load_documents(url: str) -> list:
    """Load and process web documents"""
    return SimpleWebPageReader(html_to_text=True).load_data([url])


def setup_models(config: dict[str, Any]):
    """Configure embedding and chat models"""
    
    Settings.embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
   
    llm = OpenAILike(
    api_base="https://llms-inference.innkube.fim.uni-passau.de",
    api_key=api_key,
    model="llama3.1")
    Settings.llm = llm

    Settings.transformations = [
        SentenceSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
        )
    ]



def setup_vector_store(db_path: str) -> MilvusVectorStore:
    """Initialize vector store"""
    sample_emb = Settings.embed_model.get_text_embedding("test")
    print(f"Embedding dimension: {len(sample_emb)}")
    return MilvusVectorStore(uri=db_path, dim=len(sample_emb), overwrite=True)



def create_index(documents: list, vector_store: MilvusVectorStore):
    """Create document index"""
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )



def query_document(index: VectorStoreIndex, question: str, top_k: int):
    """Query document with given question"""
    query_engine = index.as_query_engine(similarity_top_k=top_k)
    instruction = "Give a concise and a direct answer without statements like 'based on the context' or 'The text does not define'."
    response = query_engine.query(instruction + question)
    return response



def get_parser() -> argparse.ArgumentParser:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="RAG with vLLM and LlamaIndex")

    parser.add_argument(
        "--db-path", default="./milvus_demo.db", help="Path to Milvus database"
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Enable interactive Q&A mode"
    )
    parser.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size for document splitting",
    )
    parser.add_argument(
        "-o",
        "--chunk-overlap",
        type=int,
        default=100,
        help="Chunk overlap for document splitting",
    )
    parser.add_argument(
        "-k", "--top-k", type=int, default=1, help="Number of top results to retrieve"
    )
    return parser



def pipeline_for_one_document(query:str, title: str):
    
    args = get_parser().parse_args()
    
    config = init_config(args)
    setup_models(config)
    # Setup vector store
    vector_store = setup_vector_store(config["db_path"])
    filename = title_to_filename(title)
    pdf_path = f"dblp/{filename}.pdf"
    
    documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
       
    try:
        index = create_index(documents, vector_store)
        response = query_document(index, query, config["top_k"])
        return(response)
    except Exception as e:
        print(f"Querying failed for {title}: {e}")
            
    



def main():
    
    with open("ORKG_papers.json", "r") as file:
        dataset =json.load(file)
        df = pd.DataFrame(dataset)
   
    # Parse command line arguments
    args = get_parser().parse_args()

    # Initialize configuration
    config = init_config(args)

    # Setup models
    setup_models(config)

    # Setup vector store
    vector_store = setup_vector_store(config["db_path"])

    results = []

    for i, row in tqdm(df.iterrows(), desc="Generating Answers from Docs"):
        title = row["title_query"]
        #filename = title_to_filename(title)
        filename = title
        pdf_path = f"orkg_papers/papers/{filename}.pdf"
        
        if not os.path.exists(pdf_path):
            print(f"PDF not found: {pdf_path}")
            continue
        
        
        # Load document
        documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
        try:
            index = create_index(documents, vector_store)
        except Exception as e:
            print(f"Indexing failed for {title}: {e}")
            continue
        
         # Iterate over each QA pair
        for qa in row.get("qa_pairs", []):
            question = qa.get("question", None)
            answer = qa.get("answer", None)
            if not question:
                continue

            try:
                response = query_document(index, question, config["top_k"])
                results.append({
                    "title": title,
                    "question": question,
                    "response": response,
                    "ground_truth": answer
                })
                print(f"Done: {title} | Q: {question[:20]}...")
            except Exception as e:
                print(f"Querying failed for {title}, Q: {question}: {e}")
                continue
            
    # Save results (row per QA pair)
    pd.DataFrame(results).to_csv("llama_orkg_rag_responses.csv", index=False)
     


def main_old():
    

    with open("qasper_dblp_merged.json", "r") as file:
        dataset =json.load(file)
        df = pd.DataFrame(dataset)
   
    # Parse command line arguments
    args = get_parser().parse_args()

    # Initialize configuration
    config = init_config(args)

    # Setup models
    setup_models(config)

    # Setup vector store
    vector_store = setup_vector_store(config["db_path"])

    results = []
    
    for i, row in tqdm(df.iterrows(), desc="Generating Answers from Docs"):
        title = row["title"]
        filename = title_to_filename(title)
        pdf_path = f"dblp/{filename}.pdf"
        
        if not os.path.exists(pdf_path):
            print(f"PDF not found: {pdf_path}")
            continue
        
        
        # Load document
        documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
        try:
            index = create_index(documents, vector_store)
        except Exception as e:
            print(f"Indexing failed for {title}: {e}")
            continue
        
        for qa in row.get("qas", []):
            question = qa.get("question", None)
            answers = qa.get("answers", [])
            
            if not question or not answers:
                continue
            
            # Keep only answerable answers
            answer_texts = []
            for ans in answers:
                if not ans.get("unanswerable", False):
                    # Prefer free_form_answer, fallback to extractive spans
                    if ans.get("free_form_answer"):
                        answer_texts.append(ans["free_form_answer"])
                    elif ans.get("extractive_spans"):
                        answer_texts.extend(ans["extractive_spans"])
            
            # If all answers were unanswerable, skip this QA
            if not answer_texts:
                continue
            
            ground_truth = " | ".join(answer_texts)  # join multiple answers if needed

            try:
                response = query_document(index, question, config["top_k"])
                results.append({
                    "title": title,
                    "question": question,
                    "response": response,
                    "ground_truth": ground_truth
                })
                print(f"Done: {title} | Q: {question[:20]}...")
            except Exception as e:
                print(f"Querying failed for {title}, Q: {question}: {e}")
                continue
            
    pd.DataFrame(results).to_csv("llama_dblp_rag_responses.csv", index=False)
     






def route_question(question, bench):
    pattern = r'(?:from|according to|in the context of)\s+(?:the\s+)?paper\s+(?:"|“)?([A-Z][^"?]*)'
    match = re.search(pattern, question, re.IGNORECASE)
    if match:
        title = match.group(1).strip()
        print("Extracted Title:", title)
        return pipeline_for_one_document(question, title)
    
    else:
        generator = nl2sparql.SPARQLGenerator("gemma2")
        generated_query = generator.generate(
            question=question,
            templates=sim_questions,
            db=bench
            )
        query = postprocess_sparql(generated_query)
        return generator.execute(query, db=bench)





if __name__ == "__main__":
    #route_question("What is the Wikidata identifier of the author Robert S.?")
    main_old()