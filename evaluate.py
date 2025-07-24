
from sentence_transformers import SentenceTransformer, util
from bert_score import score
from rdflib.plugins.sparql.parser import parseQuery
from nltk.translate.bleu_score import sentence_bleu




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