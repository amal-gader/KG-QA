
from sentence_transformers import SentenceTransformer, util
from bert_score import score
from rdflib.plugins.sparql.parser import parseQuery
from nltk.translate.bleu_score import sentence_bleu

import re


def calculate_semantic_similarity(answer1, answer2):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb1 = model.encode(answer1, convert_to_tensor=True)
    emb2 = model.encode(answer2, convert_to_tensor=True)

    similarity = util.pytorch_cos_sim(emb1, emb2).item()
    return similarity


def extract_graph_pattern(query):
    return parseQuery(query)


def bert_score_metrics(generated_answer: str, reference_answer: str):
    P, R, F1 = score([generated_answer], [reference_answer], lang="en", verbose=True)
    return P.item(), R.item(), F1.item()


def jaccard_similarity(query1: str, query2: str):
    set1 = set(query1)
    set2 = set(query2)
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union)


def compute_bleu(reference: str, candidate: str):
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()

    bleu_score = sentence_bleu(reference_tokens, candidate_tokens)
    return bleu_score






def normalize_text(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    #s = re.sub(r'\b(a|an|the)\b', ' ', s)  # remove articles
    #s = re.sub(r'[^a-z0-9\s]', '', s)      # remove punctuation
    #s = re.sub(r'\s+', ' ', s).strip()     # remove extra whitespace
    return s

def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()
    
    common = set(pred_tokens) & set(gt_tokens)
    num_same = sum(min(pred_tokens.count(w), gt_tokens.count(w)) for w in common)
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1

