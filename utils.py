import requests
import json
from dotenv import load_dotenv
import os
import re
import codecs
import spacy

load_dotenv()
nlp = spacy.load("en_core_web_lg")
api_key = os.getenv("UNI_API_KEY")

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


# with open('template_representatives.json', 'r') as f:
#     q_templates = json.load(f)





def question_template(path: str, output_path: str):
    with open(path, 'r', encoding='utf-8') as f:
        questions = json.load(f)['questions']

    template_representatives = {}
    new_id_counter = 9
    for q in questions:
        template_id = q.get("template_id")
        if template_id is None:
            template_id = f"T0{new_id_counter}"
            new_id_counter += 1
            
        if template_id not in template_representatives:
            template_representatives[template_id] = {
                "id": q["id"],
                "question": q["question"]["string"],
                "sparql": q["query"]["sparql"]
            }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(template_representatives, f, indent=2, ensure_ascii=False)


def preprocess_text(text):
    # Decode unicode escape sequences like \\u00F3 → ó
    try:
        text = codecs.decode(text, 'unicode_escape')
    except Exception:
        pass
    #text = text.replace('\\.', '')
    text = text.replace('\\.', '')
    # Reverse name order"
    text = re.sub(r'\b([\w\-\'"]+),\s*([\w\-\'"]+\.?)', r'\2 \1', text)
    return text



    
    
def build_regex_pattern(name: str) -> str:
    name_parts = name.lower().strip().split()
    name_parts = [part.replace(".", "") for part in name_parts]
    pattern = ".*".join(name_parts)
    return f'No id is available, use regex matching, exp: ?author rdfs:label ?name .\nFILTER(REGEX(?name, "{pattern}", "i"))'
    
    


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

    
      
def title_to_filename(title: str) -> str:
    title = title.lower()
    title = re.sub(r'[^a-z0-9]+', '_', title)
    return title.strip('_')


def download_pdf(url, output_path):
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.status_code == 200 and response.headers.get("Content-Type", "").startswith("application/pdf"):
            with open(output_path, "wb") as f:
                f.write(response.content)
            return "downloaded"
        else:
            return f"invalid_pdf ({response.status_code})"
    except requests.exceptions.Timeout:
        return "timeout"
    except Exception as e:
        return f"error: {e}"

