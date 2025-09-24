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
from utils import postprocess_sparql
from transformers import pipeline

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


load_dotenv()
api_key = os.getenv("UNI_API_KEY")

client = openai.OpenAI(
    api_key=api_key,
    base_url="https://llms-inference.innkube.fim.uni-passau.de" 
    )



class SPARQLGenerator:
    
    def __init__(self, model="llama3.1", max_retries=1):
        self.model = model
        self.max_retries = max_retries
        self.delay=0.5


    def generate(self, question, templates, db="orkg", ent_link=False, link_dblp=False):
        if db == "orkg":
            prompt = self.get_orkg_prompt(question, templates)
            instruction = (
                " You are an expert in SPARQL query generation."
                " Given a natural language question, a similar question template, and its corresponding SPARQL query."
                " generate a valid SPARQL query that answers the given question."
                " If you cannot generate the query, output only: NaN. Don't include any explanation or additional text."
            )    
        elif db == "dblp":
            prompt = self.get_dblp_prompt(question, templates, ent_link, link_dblp)
            instruction = (
                "You are an expert in sparql query generation. Given a question, "
                "a similar question template, its sparql query and DBLP entity ids if available, "
                "generate a sparql query that answers the question. If you can't generate the query, "
                "return Nan. Provide only the sparql query without any explanation or additional text."
            )
        else:
            raise ValueError("Unknown DB")
        

        return self.inference(instruction, prompt)


    def execute(self, query, db="orkg"):
        if db == "orkg":
            prefix = """
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX orkgc: <http://orkg.org/orkg/class/>
            PREFIX orkgr: <http://orkg.org/orkg/resource/>
            PREFIX orkgp: <http://orkg.org/orkg/predicate/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            """
            query = prefix + '\n' + query
            url = "http://localhost:7200/repositories/orkg_kg"
        else:
            url = "http://192.168.0.163:9999/blazegraph/namespace/test/sparql"

        headers = {"Accept": "application/sparql-results+json"}

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(url, params={"query": query}, headers=headers, timeout=10)
                response.raise_for_status()  # Raise HTTPError for bad responses
                return response.json().get("results", {}).get("bindings", [])

            except requests.exceptions.RequestException as e:
                print(f"[Attempt {attempt}] Request failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.delay * attempt)  # exponential-ish backoff
                else:
                    print(f"[ERROR] Failed after {self.max_retries} attempts.")
                    return []

            except ValueError as e:
                print(f"[ERROR] JSON decoding failed: {e}")
                return []
        return []




    def get_orkg_prompt(self, question, templates):
        return "\n".join(get_similar_questions(question, templates))




    def get_dblp_prompt(self, question, templates, ent_link, link_dblp):
        if ent_link:
            return prompt_with_entity_linking(question, templates, link_dblp)
        return prompt_with_predefined_entity_ids(question, templates)
    
    
    
    def inference(self,instruction, prompt, backoff=5):
        retries=0
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
        while retries < self.max_retries:
            try: 
                
                
                # response = client.chat.completions.create(
                #     temperature=0.6,
                #     model=self.model,
                #     messages=[
                #         {"role": "system", "content": instruction },
                #         {"role": "user", "content":  prompt}
                #     ]
                # )
                # return response.choices[0].message.content
                
                
                
                # chat = ChatOpenAI(
                #             openai_api_base="https://llms-inference.innkube.fim.uni-passau.de",
                #             api_key=api_key,
                #             model = self.model,
                #             temperature=0.6
                #         )

                # messages = [
                #     SystemMessage(
                #         content=instruction
                #     ),
                #     HumanMessage(
                #         content=prompt
                #     ),
                # ]
                # response = chat(messages)
                # return response.content
                
                

                messages = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": prompt}
                ]

                inputs = tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    add_generation_prompt=True  # important! tells the model it's time to answer
                ).to(model.device)

                outputs = model.generate(
                    inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True
                )

                # Slice off the prompt tokens
                generated_tokens = outputs[0][inputs.shape[1]:]

                response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                return response

            except (RateLimitError, APIError) as e:
                print(f"[Retry {retries+1}/{self.max_retries}] Rate/API error: {e}. Retrying in {backoff} seconds...")
                time.sleep(backoff)
                retries += 1
                backoff *= 2  # Exponential backoff

            except Exception as e:
                print(f"Unexpected error during LLM call: {e}")
                raise

        print("Maximum retries exceeded. Skipping this item.")
        return None



def main(args):
    bench = args.bench
    ent_link = args.ent_link
    link_dblp = args.link_dblp
    model = args.model
    
    generator = SPARQLGenerator(model)
    
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
    for question in tqdm(questions["questions"], desc="Evaluating"):
        
        generated_query = generator.generate(
        question=question,
        templates=sim_questions,
        db=bench,
        ent_link=ent_link,
        link_dblp=link_dblp
        )
        
        processed_query=postprocess_sparql(generated_query)
        
        sparql_query = question["query"]["sparql"] 
        nl_query = question["question"]["string"]
        
        bleu = compute_bleu(sparql_query, processed_query)
        jaccard = jaccard_similarity(sparql_query, processed_query)
        bert = bert_score_metrics(sparql_query, processed_query)

        results.append({
            "Question": nl_query,
            "gt_sparql": sparql_query,
            "generated_sparql": generated_query,
            "BLEU": bleu,
            "Jaccard": jaccard,
            "BERTScore": bert,
        })
        
    df = pd.DataFrame(results)
    df.to_csv(f"{bench}_{ent_link}_{link_dblp}_{model}_nl2sparql_results.csv", index=False)
  
  

 
if __name__=='__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", default="dblp", help="The benchmark to test on")
    parser.add_argument("--ent_link", action="store_true", help="Enable entity linking")
    parser.add_argument("--model", default="gemma2", help="The model to generate the query")
    parser.add_argument("--link_dblp", action="store_true", help="Entity linking with LinkDBLP")
    args = parser.parse_args()
    main(args)   