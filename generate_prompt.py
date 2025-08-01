from sentence_transformers import util, SentenceTransformer
import json
import torch
import time
import spacy
import requests
import os
from tqdm import tqdm
from utils import ( preprocess_text, build_regex_pattern, extract_paper_titles)
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()  

nlp = spacy.load("en_core_web_lg")





def get_paper_id(paper_title):
    response = requests.get('https://dblp.org/search/publ/api', params={'q': paper_title, 'format': 'json'})
    if response.status_code == 200:
        data = response.json()
        return(data['result']['hits']['hit'][0]['info']['url'])
    else:
        print("Request failed with status:", response.status_code)
        
 
 
        
def get_author_id(author_name, max_retries=5, delay=1.0):
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(
                'https://dblp.org/search/author/api',
                params={'q': author_name, 'format': 'json'},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                if 'hit' in data['result']['hits']:
                    hits = data['result']['hits']['hit']
                    return hits[0]['info']['url'], len(hits)
                else:
                    print(f"No hits for author: {author_name}")
                    return "", 0

            elif response.status_code == 429:
                print("Rate limit hit. Sleeping and retrying...")
                time.sleep(60)  # exponential backoff
                retries += 1
            else:
                print("Request failed with status:", response.status_code)
                return "", 0

        except requests.RequestException as e:
            print("Request exception occurred:", e)
            time.sleep(delay * (2 ** retries))  
            retries += 1

    print(f"Max retries reached for author: {author_name}")
    return "", 0
        



def extract_author_dblp_ids(text):
    authors={}
    doc = nlp(text)
    nbr_ent = len(doc.ents)
    for ent in doc.ents:
        if ent.label_=='PERSON':
            print(ent.text)
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



def encode_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding.squeeze(0)




def compute_template_embeddings(q_templates):
    questions_text = [q['question']['string'] for q in q_templates['questions']]
    embeddings = torch.stack([encode_bert(q) for q in questions_text])
    return embeddings, questions_text



def find_top_similar_questions(input_embedding, template_embeddings, q_templates, top_k=2):
    similarities = F.cosine_similarity(input_embedding.unsqueeze(0), template_embeddings)
    top_indices = torch.topk(similarities, k=top_k).indices.tolist()

    results = []
    for idx in top_indices:
        q = q_templates['questions'][idx]
        results.append({
            "text": q['question']['string'],
            "sparql": q['query']['sparql']
        })
    return results




def generate_similarity_json(all_questions, q_templates, save_path="similar_questions.json"):
    template_embeddings, _ = compute_template_embeddings(q_templates)
    result_dict = {}

    for q in tqdm(all_questions):
        question_id = q['id']
        question_text = q['question']['string']
        input_embedding = encode_bert(question_text)

        similar_qs = find_top_similar_questions(
            input_embedding,
            template_embeddings,
            q_templates
        )
        result_dict[question_id] = {
            "question": question_text,
            "similar_questions": similar_qs
        }
    with open(save_path, 'w') as f:
        json.dump(result_dict, f, indent=2)





def get_similar_questions(question, similar_questions: str):
   
    question_id = question['id']
    sim_questions = similar_questions[question_id]["similar_questions"]
    nl_query = similar_questions[question_id]["question"]
    first_question, first_sparql = sim_questions[0]['text'], sim_questions[0]['sparql']
    second_question, second_sparql = sim_questions[1]['text'], sim_questions[1]['sparql']
    return [
        f"Question: {nl_query}",
        f"Similar Question 1: {first_question}",
        first_sparql,
        f"Similar Question 2: {second_question}",
        second_sparql
        ]
  

   

def prompt_with_predefined_entity_ids(question, q_templates):     
    entity_ids = question["entities"]
    promt_parts= get_similar_questions(question, q_templates)
    if entity_ids:
         promt_parts.append(f"Entity ids: {entity_ids}")
    prompt = "\n".join(promt_parts)   
    return prompt
    



def prompt_with_entity_linking(question, q_templates):
    nl_query = question["question"]["string"]
    
    prompt_parts= get_similar_questions(question, q_templates)
    
    pre_processed_question = preprocess_text(nl_query)

    author_ids = extract_author_dblp_ids(pre_processed_question)
    paper_ids = extract_paper_ids(pre_processed_question)
    

    if paper_ids:
        prompt_parts.append(f"Paper ids: {paper_ids}")
    if author_ids:
        prompt_parts.append(f"Author ids: {author_ids}")

    prompt = "\n".join(prompt_parts)
        
    return prompt



    