#!/usr/bin/env python3

import argparse
import csv
import math
import os
import re
from collections import Counter

import pymorphy3

STOP_WORDS = {
    'а', 'без', 'более', 'больше', 'будет', 'будто', 'бы', 'был', 'была',
    'были', 'было', 'быть', 'в', 'вам', 'вас', 'ведь', 'весь', 'вдруг', 'во',
    'вот', 'впрочем', 'все', 'всё', 'всего', 'всех', 'всю', 'вы', 'где', 'да',
    'даже', 'два', 'для', 'до', 'другой', 'его', 'ее', 'её', 'ей', 'ему',
    'если', 'есть', 'еще', 'ещё', 'ж', 'же', 'за', 'зачем', 'здесь', 'и',
    'из', 'или', 'им', 'иногда', 'их', 'к', 'как', 'какая', 'какой', 'когда',
    'конечно', 'кто', 'куда', 'ли', 'лучше', 'между', 'меня', 'мне', 'много',
    'мой', 'моя', 'мы', 'на', 'над', 'надо', 'наконец', 'нас', 'не', 'него',
    'нее', 'неё', 'ней', 'нельзя', 'нет', 'ни', 'нибудь', 'никогда', 'ним',
    'них', 'ничего', 'но', 'ну', 'о', 'об', 'один', 'он', 'она', 'они',
    'опять', 'от', 'перед', 'по', 'под', 'после', 'потом', 'потому', 'почти',
    'при', 'про', 'разве', 'раз', 'с', 'сам', 'свою', 'себе', 'себя',
    'сейчас', 'со', 'совсем', 'так', 'такой', 'там', 'тебя', 'тем', 'теперь',
    'то', 'тогда', 'того', 'тоже', 'только', 'том', 'тот', 'три', 'тут', 'ты',
    'у', 'уж', 'уже', 'хорошо', 'хоть', 'чего', 'чем', 'через', 'что', 'чтоб',
    'чтобы', 'чуть', 'эти', 'этих', 'это', 'этого', 'этой', 'этом', 'этот',
    'эту', 'я',
}


def load_idf(filepath):
    idf = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            idf[row[0]] = float(row[1])
    return idf


def load_processed_docs(processed_dir):
    docs = {}
    for filename in os.listdir(processed_dir):
        if not filename.endswith('.txt'):
            continue
        doc_id = int(re.search(r'\d+', filename).group())
        with open(os.path.join(processed_dir, filename), 'r', encoding='utf-8') as f:
            docs[doc_id] = f.read().split()
    return docs


def build_doc_vectors(docs, idf):
    vectors = {}
    norms = {}
    for doc_id, terms in docs.items():
        total = len(terms)
        counts = Counter(terms)
        vec = {}
        for t, c in counts.items():
            if t in idf:
                vec[t] = (c / total) * idf[t]
        norm = math.sqrt(sum(v * v for v in vec.values()))
        vectors[doc_id] = vec
        norms[doc_id] = norm
    return vectors, norms


def query_to_lemmas(query_str, morph):
    tokens = re.findall(r'[а-яёА-ЯЁ]+', query_str.lower())
    lemmas = []
    for token in tokens:
        lemma = morph.parse(token)[0].normal_form
        if lemma not in STOP_WORDS and len(lemma) > 1:
            lemmas.append(lemma)
    return lemmas


def search(query_lemmas, doc_vectors, doc_norms, idf):
    counts = Counter(query_lemmas)
    total = len(query_lemmas)
    q_vec = {}
    for t, c in counts.items():
        if t in idf:
            q_vec[t] = (c / total) * idf[t]
    if not q_vec:
        return []

    q_norm = math.sqrt(sum(v * v for v in q_vec.values()))

    results = []
    for doc_id, d_vec in doc_vectors.items():
        dot = sum(q_vec[t] * d_vec.get(t, 0) for t in q_vec)
        d_norm = doc_norms[doc_id]
        if q_norm > 0 and d_norm > 0:
            sim = round(dot / (q_norm * d_norm), 6)
            if sim > 0:
                results.append((doc_id, sim))

    results.sort(key=lambda x: (-x[1], x[0]))
    return results


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Lab 5: Vector search (cosine similarity)')
    parser.add_argument('--idf',
                        default=os.path.join(script_dir, '..', '4', 'idf.csv'),
                        help='IDF table from Lab 4')
    parser.add_argument('--processed',
                        default=os.path.join(script_dir, '..', '2', 'processed'),
                        help='Processed documents from Lab 2')
    parser.add_argument('--words-file',
                        default=os.path.join(script_dir, '..', '3', 'search_words.txt'),
                        help='File with demo words from Lab 3')
    parser.add_argument('--output',
                        default=os.path.join(script_dir, 'vector_results.csv'),
                        help='Output CSV file')
    parser.add_argument('--interactive', action='store_true',
                        help='Run interactive search mode after demo queries')
    args = parser.parse_args()

    print('Loading IDF...')
    idf = load_idf(args.idf)

    print('Loading processed documents...')
    docs = load_processed_docs(args.processed)

    print('Building document vectors...')
    doc_vectors, doc_norms = build_doc_vectors(docs, idf)

    morph = pymorphy3.MorphAnalyzer()

    if os.path.exists(args.words_file):
        with open(args.words_file, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip()]
    else:
        words = ['россия', 'история', 'культура']

    word1, word2, word3 = words[0], words[1], words[2]
    queries = [
        word1, word2, word3,
        f'{word1} {word2}',
        f'{word1} {word3}',
        f'{word2} {word3}',
        f'{word1} {word2} {word3}',
    ]

    all_results = {}
    max_rows = 0

    for query in queries:
        lemmas = query_to_lemmas(query, morph)
        results = search(lemmas, doc_vectors, doc_norms, idf)
        all_results[query] = results
        max_rows = max(max_rows, len(results))
        print(f'\nQuery "{query}" -> {len(results)} results')
        for doc_id, sim in results[:5]:
            print(f'  doc {doc_id}: {sim}')

    with open(args.output, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        header = []
        for q in queries:
            header.extend([f'{q}_doc', f'{q}_sim'])
        w.writerow(header)
        for i in range(max_rows):
            row = []
            for q in queries:
                res = all_results[q]
                if i < len(res):
                    row.extend([res[i][0], res[i][1]])
                else:
                    row.extend(['', ''])
            w.writerow(row)

    print(f'\nResults saved to {args.output}')

    if args.interactive:
        print('\n--- Interactive mode (type "exit" to quit) ---')
        while True:
            query = input('\nSearch: ').strip()
            if query.lower() == 'exit':
                break
            lemmas = query_to_lemmas(query, morph)
            if not lemmas:
                print('No valid terms in query.')
                continue
            results = search(lemmas, doc_vectors, doc_norms, idf)
            print(f'Query "{query}" (lemmas: {lemmas}) -> {len(results)} results')
            for doc_id, sim in results:
                print(f'  doc {doc_id}: {sim}')


if __name__ == '__main__':
    main()
