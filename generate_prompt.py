from utils import (get_similar_questions, preprocess_text, extract_author_dblp_ids, extract_paper_ids )

from dotenv import load_dotenv
import os
import openai
load_dotenv()
api_key = os.getenv("UNI_API_KEY")


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