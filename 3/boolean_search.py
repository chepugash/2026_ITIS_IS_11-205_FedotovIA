#!/usr/bin/env python3

import argparse
import os
import re
from collections import defaultdict

import pymorphy3


def build_inverted_index(processed_dir):
    index = defaultdict(set)
    all_docs = set()

    files = [f for f in os.listdir(processed_dir) if f.endswith('.txt')]
    for filename in files:
        doc_id = int(re.search(r'\d+', filename).group())
        all_docs.add(doc_id)
        with open(os.path.join(processed_dir, filename), 'r', encoding='utf-8') as f:
            terms = f.read().split()
        for term in set(terms):
            index[term].add(doc_id)

    return dict(index), all_docs


def save_inverted_index(index, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for term in sorted(index.keys()):
            docs_str = ' '.join(str(d) for d in sorted(index[term]))
            f.write(f'{term} {docs_str}\n')


def pick_demo_words(index, all_docs, count=3):
    n = len(all_docs)
    candidates = []
    for term, docs in index.items():
        frac = len(docs) / n
        if 0.15 <= frac <= 0.70:
            candidates.append(term)
    candidates.sort(key=lambda t: -len(index[t]))

    for i, w1 in enumerate(candidates):
        for j, w2 in enumerate(candidates):
            if j <= i:
                continue
            for k, w3 in enumerate(candidates):
                if k <= j:
                    continue
                if index[w1] & index[w2] & index[w3]:
                    return [w1, w2, w3]

    fallback = sorted(index.keys(), key=lambda t: -len(index[t]))
    return fallback[:count]


class BooleanSearcher:
    def __init__(self, index, all_docs):
        self.index = index
        self.all_docs = all_docs
        self.morph = pymorphy3.MorphAnalyzer()

    def search(self, query):
        normalized = re.sub(r'\bИЛИ\b', '|', query)
        normalized = re.sub(r'\bНЕ\b', '!', normalized)
        normalized = re.sub(r'\bИ\b', '&', normalized)
        self.tokens = self._lex(normalized)
        self.pos = 0
        result = self._parse_or()
        return result

    def _lex(self, query):
        tokens = []
        i = 0
        while i < len(query):
            ch = query[i]
            if ch in '&|!()':
                tokens.append(ch)
                i += 1
            elif ch.isspace():
                i += 1
            else:
                j = i
                while j < len(query) and query[j] not in '&|!() \t':
                    j += 1
                word = query[i:j].lower()
                lemma = self.morph.parse(word)[0].normal_form
                tokens.append(('WORD', lemma))
                i = j
        return tokens

    def _cur(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _eat(self, expected=None):
        tok = self._cur()
        if expected is not None and tok != expected:
            raise ValueError(f'Expected {expected!r}, got {tok!r}')
        self.pos += 1
        return tok

    def _parse_or(self):
        result = self._parse_and()
        while self._cur() == '|':
            self._eat('|')
            result = result | self._parse_and()
        return result

    def _parse_and(self):
        result = self._parse_not()
        while self._cur() == '&':
            self._eat('&')
            result = result & self._parse_not()
        return result

    def _parse_not(self):
        if self._cur() == '!':
            self._eat('!')
            return self.all_docs - self._parse_not()
        return self._parse_primary()

    def _parse_primary(self):
        tok = self._cur()
        if tok == '(':
            self._eat('(')
            result = self._parse_or()
            self._eat(')')
            return result
        if isinstance(tok, tuple) and tok[0] == 'WORD':
            self._eat()
            return set(self.index.get(tok[1], set()))
        raise ValueError(f'Unexpected token: {tok!r}')


def main():
    parser = argparse.ArgumentParser(description='Lab 3: Inverted index & boolean search')
    parser.add_argument('--input',
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', '2', 'processed'),
                        help='Processed documents directory')
    parser.add_argument('--output',
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'inverted_index.txt'),
                        help='Output inverted index file')
    parser.add_argument('--words-file',
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'search_words.txt'),
                        help='File to save chosen demo words')
    parser.add_argument('--interactive', action='store_true',
                        help='Run interactive search mode after demo queries')
    args = parser.parse_args()

    print('Building inverted index...')
    index, all_docs = build_inverted_index(args.input)
    save_inverted_index(index, args.output)
    print(f'Index saved: {len(index)} terms, {len(all_docs)} documents\n')

    words = pick_demo_words(index, all_docs)
    word1, word2, word3 = words[0], words[1], words[2]

    with open(args.words_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(words))
    print(f'Demo words (saved to {args.words_file}): {word1}, {word2}, {word3}')

    searcher = BooleanSearcher(index, all_docs)

    queries = [
        f'{word1} & {word2} & {word3}',
        f'{word1} & {word2} & !{word3}',
        f'{word1} & {word2} | {word3}',
        f'{word1} & !{word2} | !{word3}',
        f'{word1} | {word2} | {word3}',
    ]

    print('=' * 70)
    for query in queries:
        result = searcher.search(query)
        docs = sorted(result)
        print(f'\nQuery: {query}')
        print(f'Documents ({len(docs)}): {docs}')

    if args.interactive:
        print('\n--- Interactive mode (type "exit" to quit) ---')
        while True:
            query = input('\nQuery: ').strip()
            if query.lower() == 'exit':
                break
            try:
                result = searcher.search(query)
                print(f'Documents ({len(result)}): {sorted(result)}')
            except Exception as e:
                print(f'Error: {e}')


if __name__ == '__main__':
    main()
