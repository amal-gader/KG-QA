import openai
from sentence_transformers import util, SentenceTransformer
import json
import torch
from dotenv import load_dotenv
import os
import spacy
import requests

from utils import ( preprocess_text, build_regex_pattern, extract_paper_titles)

load_dotenv()
nlp = spacy.load("en_core_web_lg")
api_key = os.getenv("UNI_API_KEY")



with open('DBLP-QuAD/DBLP-QuAD/template_representatives.json', 'r') as f:
    q_templates = json.load(f)



def get_paper_id(paper_title):
    response = requests.get('https://dblp.org/search/publ/api', params={'q': paper_title, 'format': 'json'})
    if response.status_code == 200:
        data = response.json()
        return(data['result']['hits']['hit'][0]['info']['url'])
    else:
        print("Request failed with status:", response.status_code)
        
        
def get_author_id(author_name):
    response = requests.get('https://dblp.org/search/author/api', params={'q': author_name, 'format': 'json'})
    if response.status_code == 200:
        data = response.json()
        return(data['result']['hits']['hit'][0]['info']['url'], len(data['result']['hits']['hit']))
    else:
        print("Request failed with status:", response.status_code)
        return None
            


def extract_author_dblp_ids(text):
    authors={}
    doc = nlp(text)
    nbr_ent = len(doc.ents)
    for ent in doc.ents:
        if ent.label_=='PERSON':
            match, nbr_matches = get_author_id(ent.text)
            if nbr_matches>1 and nbr_ent>1:
                authors[ent.text]=build_regex_pattern(ent.text)
            else:
                authors[ent.text]=match
    return authors



def extract_paper_ids(question):
    papers = extract_paper_titles(question)
    paper_ids = {}
    if papers:
        for paper in papers:
            paper_ids[paper]=get_paper_id(paper)
            return paper_ids
    else:
        return None


def get_similar_questions(question, q_templates):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    questions_text = [q['question'] for q in list(q_templates.values())]
    embeddings = model.encode(questions_text, convert_to_tensor=True)
    new_embedding = model.encode(question, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(new_embedding, embeddings)[0]

    # Get bottom 2 indices (least similar)
    top_indices = torch.topk(similarities, k=2).indices.tolist()

    results = []
    for idx in top_indices:
        q = list(q_templates.values())[idx]
        results.append((q['question'], q['sparql']))

    return results




def prompt(question, q_templates):
    
    first_similar_question, second_similar_question = get_similar_questions(question, q_templates)
    first_question, first_sparql = first_similar_question
    second_question, second_sparql = second_similar_question
    pre_processed_question = preprocess_text(question)
    author_ids = extract_author_dblp_ids(pre_processed_question)
    paper_ids = extract_paper_ids(pre_processed_question)
    
    prompt = f"Question: {question}\nSimilar Question 1: {first_question}\n{first_sparql}\nSimilar Question 2: {second_question}\n{second_sparql}\nPaper ids: {paper_ids}\nAuthor ids: {author_ids}"
    return prompt



def nl2sparql(question, templates, model="qwen2.5"):
    prompt_text = prompt(question, templates)
    
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