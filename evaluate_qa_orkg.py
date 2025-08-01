import pandas as pd
from utils import extract_value_tuples, postprocess_sparql
from nl2sparql import execute_query_orkg

from tqdm import tqdm
from collections import Counter

tqdm.pandas()





def evaluate_multiset_results(bindings1, bindings2):
    tuples1 = extract_value_tuples(bindings1)
    tuples2 = extract_value_tuples(bindings2)

    counter1 = Counter(tuples1)
    counter2 = Counter(tuples2)

    intersection = counter1 & counter2
    union = counter1 | counter2

    tp = sum(intersection.values())  
    total_pred = sum(counter2.values())
    total_gold = sum(counter1.values())

    precision = tp / total_pred if total_pred else 0
    recall = tp / total_gold if total_gold else 0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0
    exact_match = counter1 == counter2

    jaccard = sum(intersection.values()) / sum(union.values()) if union else 1.0

    return {
        "exact_match": exact_match,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "jaccard": jaccard
    }



df = pd.read_csv("orkg_nl2sparql_gemma_results.csv")

df["Generated SPARQL"] = df["Generated SPARQL"].apply(postprocess_sparql)

df["answer"] = df["Generated SPARQL"].progress_apply(execute_query_orkg)
df["gt_answer"] = df["GT SPARQL"].progress_apply(execute_query_orkg)
df["metrics"] = df.progress_apply(lambda x: evaluate_multiset_results(x["answer"], x["gt_answer"]), axis=1)




df.to_csv("qa_results.csv")



    