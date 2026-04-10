#!/usr/bin/env python3

import argparse
import os
import re

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


def tokenize(text):
    return re.findall(r'[а-яёА-ЯЁ]+', text.lower())


def process_document(text, morph):
    tokens = tokenize(text)
    lemmas = []
    for token in tokens:
        lemma = morph.parse(token)[0].normal_form
        if lemma not in STOP_WORDS and len(lemma) > 1:
            lemmas.append(lemma)
    return lemmas


def main():
    parser = argparse.ArgumentParser(description='Lab 2: Tokenize and lemmatize documents')
    parser.add_argument('--input',
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', '1', 'pages'),
                        help='Input pages directory (default: ../1/pages)')
    parser.add_argument('--output',
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'processed'),
                        help='Output processed directory (default: ./processed)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    morph = pymorphy3.MorphAnalyzer()

    files = sorted(
        [f for f in os.listdir(args.input) if f.endswith('.txt')],
        key=lambda x: int(re.search(r'\d+', x).group()),
    )

    for filename in files:
        with open(os.path.join(args.input, filename), 'r', encoding='utf-8') as f:
            text = f.read()

        lemmas = process_document(text, morph)

        with open(os.path.join(args.output, filename), 'w', encoding='utf-8') as f:
            f.write(' '.join(lemmas))

        print(f'[OK] {filename}: {len(lemmas)} lemmas')

    print(f'\nDone: processed {len(files)} documents -> {args.output}')


if __name__ == '__main__':
    main()
