from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import openai
import requests
from sentence_transformers import util, SentenceTransformer
import json
import torch
from dotenv import load_dotenv
import os
import re
import codecs
import spacy

load_dotenv()
nlp = spacy.load("en_core_web_lg")
api_key = os.getenv("UNI_API_KEY")



with open('template_representatives.json', 'r') as f:
    q_templates = json.load(f)




def get_similar_question(question, corpus):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    questions_text = [q['question'] for q in list(corpus.values())]
    embeddings = model.encode(questions_text, convert_to_tensor=True)
    question_embedding = model.encode(question, convert_to_tensor=True)
    
    hits = util.semantic_search(question_embedding, embeddings)
    hits = hits[0]  
    for hit in hits[0:5]:
        print("\t{:.3f}\t{}".format(hit['score'], questions_text[hit['corpus_id']]))
    return [questions_text[hit['corpus_id']] for hit in hits[0:2]]



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



def get_similar_question_embeddings(question, corpus, model):
    embeddings = OpenAIEmbeddings(
    openai_api_base="https://llms-inference.innkube.fim.uni-passau.de",
    api_key=api_key,
    model=model
    )
    questions_text = [q['question'] for q in list(corpus.values())]
    corpus_embeddings = [embeddings.embed_query(q) for q in questions_text]
    query_embedding = embeddings.embed_query(question)
    cos_sim = cosine_similarity([query_embedding], corpus_embeddings)[0]
    top_n = sorted(enumerate(cos_sim), key=lambda x: x[1], reverse=True)[:2]
   
    return [(list(corpus.values())[idx]['question'], list(corpus.values())[idx]['sparql']) for idx, score in top_n]



def preprocess_text(text):
    # Decode unicode escape sequences like \\u00F3 → ó
    try:
        text = codecs.decode(text, 'unicode_escape')
    except Exception:
        pass  # Fail silently if already decoded
    # Remove escaped periods like '\.' → '.'
    text = text.replace('\\.', '')
    # Remove comma between two name-like words
    text = re.sub(r'(\b[\w\-\']+\b),\s*(\b[\w\-\']+\b)', r'\1 \2', text)
    return text


def get_author_id(author_name):
    response = requests.get('https://dblp.org/search/author/api', params={'q': author_name, 'format': 'json'})
    if response.status_code == 200:
        data = response.json()
        return(data['result']['hits']['hit'][0]['info']['url'], len(data['result']['hits']['hit']))
    else:
        print("Request failed with status:", response.status_code)
        return None
    
    
    
def build_regex_pattern(name: str) -> str:
    name_parts = name.lower().strip().split()
    name_parts = [part.replace(".", "") for part in name_parts]
    pattern = ".*".join(name_parts)
    return f'No id is available, use regex matching, exp: ?author rdfs:label ?name .\nFILTER(REGEX(?name, "{pattern}", "i"))'
    
    



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




def extract_paper_titles(text):
    patterns = [
        r"'([^']+)'.*?'([^']+)'.*?which one.*?(?:published|authors)|which one.*?(?:published|authors).*?'([^']+)'.*?'([^']+)'",
        r"(?:\bpaper\s+|\bauthors\s+of\s+)'([^']+)'"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return [g for g in match.groups() if g is not None]
    return None




def get_paper_id(paper_title):
    response = requests.get('https://dblp.org/search/publ/api', params={'q': paper_title, 'format': 'json'})
    if response.status_code == 200:
        data = response.json()
        return(data['result']['hits']['hit'][0]['info']['url'])
    else:
        print("Request failed with status:", response.status_code)
        
        
        
def extract_paper_ids(question):
    papers = extract_paper_titles(question)
    paper_ids = {}
    if papers:
        for paper in papers:
            paper_ids[paper]=get_paper_id(paper)
            return paper_ids
    else:
        return None
    
    
    
      


