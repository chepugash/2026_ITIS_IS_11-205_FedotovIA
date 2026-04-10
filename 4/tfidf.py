#!/usr/bin/env python3

import argparse
import csv
import math
import os
import re
from collections import Counter, defaultdict


def load_processed_docs(processed_dir):
    docs = {}
    files = [f for f in os.listdir(processed_dir) if f.endswith('.txt')]
    for filename in files:
        doc_id = int(re.search(r'\d+', filename).group())
        with open(os.path.join(processed_dir, filename), 'r', encoding='utf-8') as f:
            docs[doc_id] = f.read().split()
    return docs


def compute_tf(docs):
    tf = {}
    for doc_id, terms in docs.items():
        total = len(terms)
        counts = Counter(terms)
        tf[doc_id] = {t: round(c / total, 6) for t, c in counts.items()}
    return tf


def compute_idf(docs):
    n = len(docs)
    df = defaultdict(int)
    for terms in docs.values():
        for term in set(terms):
            df[term] += 1
    return {t: round(math.log(n / freq), 6) for t, freq in df.items()}


def compute_tfidf(tf, idf):
    tfidf = {}
    for doc_id, tf_doc in tf.items():
        tfidf[doc_id] = {
            t: round(v * idf.get(t, 0), 6)
            for t, v in tf_doc.items()
        }
    return tfidf


def save_matrix(data, all_terms, doc_ids, filepath):
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['doc_id'] + list(all_terms))
        for doc_id in doc_ids:
            row = [doc_id]
            doc_data = data.get(doc_id, {})
            row.extend(doc_data.get(t, 0) for t in all_terms)
            w.writerow(row)


def save_idf_table(idf, filepath):
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['term', 'idf'])
        for term in sorted(idf.keys()):
            w.writerow([term, idf[term]])


def main():
    parser = argparse.ArgumentParser(description='Lab 4: TF / IDF / TF-IDF')
    parser.add_argument('--input',
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', '2', 'processed'),
                        help='Processed documents directory')
    parser.add_argument('--output',
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help='Output directory for CSV tables')
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    print('Loading documents...')
    docs = load_processed_docs(args.input)
    doc_ids = sorted(docs.keys())

    print('Computing TF...')
    tf = compute_tf(docs)

    print('Computing IDF...')
    idf = compute_idf(docs)

    print('Computing TF-IDF...')
    tfidf = compute_tfidf(tf, idf)

    all_terms = sorted(idf.keys())
    print(f'Vocabulary: {len(all_terms)} terms, {len(doc_ids)} documents')

    save_idf_table(idf, os.path.join(args.output, 'idf.csv'))
    print('  idf.csv saved')

    save_matrix(tf, all_terms, doc_ids, os.path.join(args.output, 'tf.csv'))
    print('  tf.csv saved')

    save_matrix(tfidf, all_terms, doc_ids, os.path.join(args.output, 'tfidf.csv'))
    print('  tfidf.csv saved')

    print(f'\nDone. All tables in {args.output}/')


if __name__ == '__main__':
    main()
