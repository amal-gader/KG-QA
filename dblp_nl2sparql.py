import json
import requests
from dotenv import load_dotenv
import os
import openai
import pandas as pd
from tqdm import tqdm

from generate_prompt import prompt_with_entity_linking, prompt_with_predefined_entity_ids
from evaluate import bert_score_metrics, compute_bleu, jaccard_similarity

load_dotenv()
api_key = os.getenv("UNI_API_KEY")


def nl2sparql(question, templates,ent_link=False, entity_ids=None, model="llama3.1"):
    
    if ent_link:
        prompt_text = prompt_with_entity_linking(question, templates)
    else: 
        prompt_text = prompt_with_predefined_entity_ids(question, templates, entity_ids)
    
    client = openai.OpenAI(
    api_key=api_key,
    base_url="https://llms-inference.innkube.fim.uni-passau.de" 
    )
    
    
    instruction = """You are an expert in sparql query generation. Given a question, a similar question template, 
    its sparql query and DBLP entity ids if available, generate a sparql query that answers the question. If you can't generate the query
    return Nan. Provide only the sparql query without any explanation or additional text."""

    response = client.chat.completions.create(
        temperature=0.6,
        model=model,
        messages = [
            {
                "role": "system",
                "content" : instruction
            },
            {   
                "role": "user",
                "content": prompt_text
            }
        ]
    )
    
    return response.choices[0].message.content  




def execute_query(query):
    url = "localhost:9999/blazegraph/namespace/kb/sparql"
    response = requests.get(url, params={'query': query}, headers={"Accept": "application/sparql-results+json"}, verify=False)
    try:
        data = response.json()
        for answer in data['results']['bindings']:
            print(answer['answer']['value']) 
        return [answer['answer']['value'] for answer in data['results']['bindings']]
    except Exception as e:
        print("Failed to parse JSON:", e)
        




def main(args):
    ent_link = args.ent_link
    with open("DBLP-QuAD/DBLP-QuAD/test_template_representatives.json", 'r') as f:
        q_templates = json.load(f)
    
    with open("DBLP-QuAD/DBLP-QuAD/test/questions.json", "r") as file:
        questions = json.load(file)
    
    results = []
    for question in tqdm(questions["questions"], desc="Evaluating"):
        sparql_query = question["query"]["sparql"] 
        nl_query = question["paraphrased_question"]["string"]
        entity_ids = question["entities"]
        generated_query = nl2sparql(nl_query,q_templates,ent_link=ent_link, entity_ids=entity_ids)
      
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
    df.to_csv("dblp_nl2sparql_results.csv", index=False)
  
  
        

if __name__=='__main__' :
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ent_link", action="store_true", help="Enable entity linking")
    args = parser.parse_args()
    main(args)   