import argparse
import pandas as pd
from utils import extract_value_tuples, postprocess_sparql
from nl2sparql import SPARQLGenerator

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



if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='gemma2')
    parser.add_argument("--bench", default="dblp")
    args=parser.parse_args()
    
    model=args.model
    bench=args.bench

   
    generator = SPARQLGenerator(model)

    df = pd.read_csv(f"results/{bench}_{model}_nl2sparql_results.csv")

    df.assign(
        **{
            "generated_sparql": lambda df: df["generated_sparql"].progress_apply(postprocess_sparql),
            "answer": lambda df: df["generated_sparql"].progress_apply(lambda x: generator.execute(x, db=bench)),
            "gt_answer": lambda df: df["gt_sparql"].progress_apply(lambda x: generator.execute(x, db=bench)),
            "metrics": lambda df: df.progress_apply(lambda x: evaluate_multiset_results(x["answer"], x["gt_answer"]), axis=1)   
        })


    df.to_csv(f"{bench}_{model}_qa_results.csv")



    