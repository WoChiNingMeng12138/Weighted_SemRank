import json
import math
from collections import defaultdict

input_file = './LitSearch/specter2_corpus_with-topic-terms.json'
output_file = './LitSearch/idf_weights.json'

with open(input_file, 'r', encoding='utf-8') as f:
    corpus_data = json.load(f)

df_counts = defaultdict(int)
valid_doc_count = 0

for corpusid, info in corpus_data.items():
    topic_names = [t[1].strip().lower() for t in info.get('topics', []) if isinstance(t, (list, tuple)) and len(t) >= 2 and str(t[1]).strip() != '']
    terms = [str(term).strip().lower() for term in info.get('terms', []) if str(term).strip() != '']

    unique_concepts = set(topic_names + terms)
    if not unique_concepts:
        continue

    valid_doc_count += 1
    for concept in unique_concepts:
        df_counts[concept] += 1

idf_weights = {}
for concept, df in df_counts.items():
    idf = math.log(1.0 + (valid_doc_count - df + 0.5)/(df + 0.5)) # BM25's idf fomula
    idf_weights[concept] = idf

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(idf_weights, f, indent=4, ensure_ascii=False)

print(f"There are {len(idf_weights)} unique concepts")
print(f"Processed {valid_doc_count} valid documents")