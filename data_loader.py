import os
import time
import requests
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

tqdm.pandas()

from utils import title_to_filename, download_pdf

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}
OUTPUT_DIR = "pdfs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 500
SEMANTIC_BATCH_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/batch"
FIELDS = "title,isOpenAccess,openAccessPdf"





def get_paper_id_from_title(title):
    url = "https://api.semanticscholar.org/graph/v1/paper/search/match"
    params = {
        "query": title,
        "fields": "paperId,title,url"
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        paper = response.json()
        return paper['data'][0].get("paperId")
    
    else:
        print("Error:", response.status_code, response.text)
        return None



def get_pdf_url_from_dblp(title):
    try:
        url = f"https://dblp.org/search/publ/api?q={requests.utils.quote(title)}&format=json"
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            return None, "dblp_error"

        data = response.json()
        hits = data.get("result", {}).get("hits", {}).get("hit", [])

        for hit in hits:
            info = hit.get("info", {})
            ee = info.get("ee", "")
            access = info.get("access", "")
            if access == "open" or ee:
                return ee, "dblp"
        return None, "dblp_no_url"
    except Exception as e:
        return None, f"dblp_fail: {e}"


def batch_fetch_metadata(paper_ids):
    all_metadata = []
    for i in tqdm(range(0, len(paper_ids), BATCH_SIZE), desc="Batch fetching"):
        batch_ids = paper_ids[i:i + BATCH_SIZE]
        try:
            response = requests.post(
                SEMANTIC_BATCH_ENDPOINT + f"?fields={FIELDS}",
                json={"ids": batch_ids},
                headers=HEADERS,
                timeout=30
            )
            if response.status_code == 200:
                all_metadata.extend(response.json())
            else:
                print(f"Batch request failed: {response.status_code} — {response.text}")
        except Exception as e:
            print(f"Exception during batch fetch: {e}")
        time.sleep(1)
    return all_metadata


def run():
    print("Loading QASPER dataset...")
    dataset = load_dataset("allenai/qasper")
    df = dataset["train"].to_pandas()
    
    df['paperId'] = df['title'].progress_apply(lambda x: get_paper_id_from_title(x))

    print("Extracting paper IDs from metadata...")
    paper_ids = df["paperId"].dropna().unique().tolist()

    print("Fetching metadata from Semantic Scholar...")
    papers = batch_fetch_metadata(paper_ids)

    results = []
    print("Processing and downloading PDFs...")
    for paper in tqdm(papers):
        title = paper.get("title")
        paper_id = paper.get("paperId")
        isOpenAccess = paper.get("isOpenAccess", False)
        pdf_url = paper.get("openAccessPdf", {}).get("url")
        license = paper.get("openAccessPdf", {}).get("license")
        source = "semantic-scholar"

        filename = title_to_filename(title)
        output_path = os.path.join(OUTPUT_DIR, filename + ".pdf")

        if pdf_url:
            status = download_pdf(pdf_url, output_path)
        else:
            fallback_url, source = get_pdf_url_from_dblp(title)
            pdf_url = fallback_url
            status = "fallback_only (not_downloaded)"

        results.append({
            "title": title,
            "paperId": paper_id,
            "pdf_url": pdf_url,
            "filename": filename,
            "status": status,
            "source": source,
            "license": license,
            "isOpenAccess": isOpenAccess
        })

    # Save 
    results_df = pd.DataFrame(results)
    results_df.to_csv("paper_metadata.csv", index=False)
    print("Done. Results saved to: paper_metadata.csv")


if __name__ == "__main__":
    run()
