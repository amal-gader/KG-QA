import argparse
import json
import requests
from dotenv import load_dotenv
import os
import openai
import pandas as pd
from tqdm import tqdm
import time
from openai import RateLimitError, APIError

from generate_prompt import prompt_with_entity_linking, prompt_with_predefined_entity_ids, get_similar_questions
from evaluate import bert_score_metrics, compute_bleu, jaccard_similarity
from utils import check_sparql

load_dotenv()
api_key = os.getenv("UNI_API_KEY")

client = openai.OpenAI(
    api_key=api_key,
    base_url="https://llms-inference.innkube.fim.uni-passau.de" 
    )



def nl2sparql_orkg(question, templates, model="qwen2.5", max_retries=5):
    prompt = get_similar_questions(question, templates)
    
    instruction = """You are an expert in sparql query generation. Given a question, a similar question template, 
    its sparql query, generate a sparql query that answers the question. If you can't generate the query
    return Nan. Provide only the sparql query without any explanation or additional text."""
    
    retries = 0
    backoff = 5  
    
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                temperature=0.6,
                model=model,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content

        except (RateLimitError, APIError) as e:
            print(f"[Retry {retries+1}/{max_retries}] Rate/API error: {e}. Retrying in {backoff} seconds...")
            time.sleep(backoff)
            retries += 1
            backoff *= 2  # Exponential backoff

        except Exception as e:
            print(f"Unexpected error during LLM call: {e}")
            raise

    raise RuntimeError("Maximum retries exceeded while calling LLM API.")
    
    



def nl2sparql(question, templates, ent_link=False, model="qwen2.5", max_retries=5):
    if ent_link:
        prompt_text = prompt_with_entity_linking(question, templates)
    else:
        prompt_text = prompt_with_predefined_entity_ids(question, templates)

    instruction = """You are an expert in sparql query generation. Given a question, a similar question template, 
    its sparql query and DBLP entity ids if available, generate a sparql query that answers the question. If you can't generate the query,
    return Nan. Provide only the sparql query without any explanation or additional text."""

    retries = 0
    backoff = 5  

    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                temperature=0.6,
                model=model,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": prompt_text}
                ]
            )
            return response.choices[0].message.content

        except (RateLimitError, APIError) as e:
            print(f"[Retry {retries+1}/{max_retries}] Rate/API error: {e}. Retrying in {backoff} seconds...")
            time.sleep(backoff)
            retries += 1
            backoff *= 2  # Exponential backoff

        except Exception as e:
            print(f"Unexpected error during LLM call: {e}")
            raise

    raise RuntimeError("Maximum retries exceeded while calling LLM API.")




def execute_query(query):
    
    if check_sparql(query):
        url = "localhost:9999/blazegraph/namespace/kb/sparql"
        response = requests.get(
            url, params={'query': query},
            headers={"Accept": "application/sparql-results+json"},
            verify=False
            )
        
        try:
            data = response.json()
            for answer in data['results']['bindings']:
                print(answer['answer']['value']) 
            return [answer['answer']['value'] for answer in data['results']['bindings']]
        except Exception as e:
            print("Failed to parse JSON:", e)
    else:
        print("Invalid SPARQL query:", query)
        return None





def execute_query_orkg(query):
    
    if check_sparql(query):
        prefix = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX orkgc: <http://orkg.org/orkg/class/>
        PREFIX orkgp: <http://orkg.org/orkg/predicate/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> """
        
        full_query = prefix + '\n' + query
        url =  "http://localhost:7200/repositories/dblp_kg"
        
        headers = {
        "Accept": "application/sparql-results+json"}
        
        response = requests.get(url, params={'query': full_query}, headers=headers)

        if response.status_code == 200:
            try:
                data = response.json()
                for binding in data["results"]["bindings"]:
                    print(binding)
                return [binding['answer']['value'] for binding in data["results"]["bindings"]]
            except Exception as e:
                print("Failed to parse JSON:", e)
        else:
            print(f"Error {response.status_code}: {response.text}")
    else:
        print("Invalid SPARQL query:", query)
        return None



def main(args):
    bench = args.bench
    ent_link = args.ent_link
    
    if bench=="orkg":
        with open("SciQA-dataset/SciQA-dataset/similar_questions.json", 'r') as f:
            sim_questions = json.load(f)
    
        with open("SciQA-dataset/SciQA-dataset/test/questions.json", "r") as file:
            questions = json.load(file)
    else:  
        with open("DBLP-QuAD/DBLP-QuAD/similar_questions.json", 'r') as f:
            sim_questions = json.load(f)
            
        with open("DBLP-QuAD/DBLP-QuAD/test/questions.json", "r") as file:
            questions = json.load(file)
     
    results = []
    for question in tqdm(questions["questions"][:500], desc="Evaluating"):
        
        
        #generated_query = nl2sparql(question,sim_questions,ent_link=ent_link)
        generated_query = nl2sparql_orkg(question,sim_questions)
        
        sparql_query = question["query"]["sparql"] 
        nl_query = question["question"]["string"]
        
        
        bleu = compute_bleu(sparql_query, generated_query)
        jaccard = jaccard_similarity(sparql_query, generated_query)
        bert = bert_score_metrics(sparql_query, generated_query)
        #sentence_sim = calculate_semantic_similarity(sparql_query, generated_query)

        results.append({
            "Question": nl_query,
            "GT SPARQL": sparql_query,
            "Generated SPARQL": generated_query,
            "BLEU": bleu,
            "Jaccard": jaccard,
            "BERTScore": bert,
            #"SentenceSimilarity": sentence_sim
        })
        
    df = pd.DataFrame(results)
    df.to_csv(f"{bench}_nl2sparql_results.csv", index=False)
  
  

 
if __name__=='__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", default="dblp", help="The benchmark to test on")
    parser.add_argument("--ent_link", action="store_true", help="Enable entity linking")
    args = parser.parse_args()
    main(args)   