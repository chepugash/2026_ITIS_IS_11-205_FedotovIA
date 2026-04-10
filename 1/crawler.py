#!/usr/bin/env python3

import argparse
import os
import re
import time
from collections import deque
from urllib.parse import urlparse, urlunparse, urljoin, quote, unquote

import requests
from bs4 import BeautifulSoup


def normalize_url(url):
    parsed = urlparse(url)
    try:
        host = parsed.hostname.encode('idna').decode('ascii')
    except (UnicodeError, UnicodeDecodeError, AttributeError):
        host = parsed.hostname or ''

    if parsed.port and parsed.port != {'http': 80, 'https': 443}.get(parsed.scheme):
        netloc = f'{host}:{parsed.port}'
    else:
        netloc = host

    path = quote(unquote(parsed.path), safe='/:@!$&\'()*+,;=-._~')
    query = quote(unquote(parsed.query), safe='=&/:@!$\'()*+,;-._~')
    return urlunparse((parsed.scheme, netloc, path, '', query, ''))


def readable_url(url):
    parsed = urlparse(url)
    host = parsed.hostname or ''
    try:
        host = host.encode('ascii').decode('idna')
    except (UnicodeError, UnicodeDecodeError):
        pass

    if parsed.port and parsed.port != {'http': 80, 'https': 443}.get(parsed.scheme):
        netloc = f'{host}:{parsed.port}'
    else:
        netloc = host

    path = unquote(parsed.path)
    query = unquote(parsed.query)
    return urlunparse((parsed.scheme, netloc, path, '', query, ''))


def extract_text(html):
    soup = BeautifulSoup(html, 'lxml')
    for tag in soup(['script', 'style', 'nav', 'header', 'footer',
                     'noscript', 'aside', 'meta', 'link',
                     'pre', 'code', 'svg', 'figure', 'table']):
        tag.decompose()
    text = soup.get_text(separator=' ', strip=True)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_links(html, base_url):
    soup = BeautifulSoup(html, 'lxml')
    links = set()
    for a in soup.find_all('a', href=True):
        absolute = urljoin(base_url, a['href'])
        normalized = normalize_url(absolute)
        links.add(normalized)
    return links


def is_same_domain(url, allowed_domains):
    parsed = urlparse(url)
    host = parsed.hostname or ''
    try:
        host = host.encode('idna').decode('ascii')
    except (UnicodeError, UnicodeDecodeError):
        pass
    return any(host == d or host.endswith('.' + d) for d in allowed_domains)


def has_cyrillic(text, threshold=0.3):
    alpha = [ch for ch in text if ch.isalpha()]
    if len(alpha) < 100:
        return False
    cyrillic = sum(1 for ch in alpha if '\u0400' <= ch <= '\u04ff')
    return cyrillic / len(alpha) >= threshold


def crawl(seed_urls, output_dir, min_pages, min_words, delay):
    pages_dir = os.path.join(output_dir, 'pages')
    os.makedirs(pages_dir, exist_ok=True)

    allowed_domains = set()
    for url in seed_urls:
        host = urlparse(url).hostname or ''
        try:
            host = host.encode('idna').decode('ascii')
        except (UnicodeError, UnicodeDecodeError):
            pass
        allowed_domains.add(host)

    queue = deque()
    visited = set()
    index = []
    doc_id = 1

    for url in seed_urls:
        n = normalize_url(url)
        if n not in visited:
            queue.append(n)
            visited.add(n)

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/124.0.0.0 Safari/537.36',
        'Accept-Language': 'ru-RU,ru;q=0.9,en;q=0.1',
    })

    while queue and len(index) < min_pages:
        url = queue.popleft()

        try:
            resp = session.get(url, timeout=15, allow_redirects=True)
            resp.raise_for_status()
        except Exception as e:
            print(f'[SKIP] {url}: {e}')
            continue

        content_type = resp.headers.get('Content-Type', '')
        if 'text/html' not in content_type:
            continue

        if resp.encoding and resp.encoding.lower() in ('iso-8859-1', 'latin-1'):
            resp.encoding = resp.apparent_encoding

        html = resp.text
        text = extract_text(html)
        wc = len(text.split())

        if wc < min_words:
            print(f'[SKIP] {url}: {wc} words < {min_words}')
        elif not has_cyrillic(text):
            print(f'[SKIP] {url}: not enough Cyrillic content')
        else:
            filepath = os.path.join(pages_dir, f'{doc_id}.txt')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)
            display_url = readable_url(url)
            index.append((doc_id, display_url))
            print(f'[OK {doc_id}/{min_pages}] {display_url} ({wc} words)')
            doc_id += 1

        try:
            for link in extract_links(html, url):
                if link not in visited and is_same_domain(link, allowed_domains):
                    parsed = urlparse(link)
                    if parsed.scheme in ('http', 'https'):
                        visited.add(link)
                        queue.append(link)
        except Exception:
            pass

        time.sleep(delay)

    index_path = os.path.join(output_dir, 'index.txt')
    with open(index_path, 'w', encoding='utf-8') as f:
        for did, url in index:
            f.write(f'{did}\t{url}\n')

    print(f'\nDone: {len(index)} pages saved to {pages_dir}')
    print(f'Index: {index_path}')


def main():
    parser = argparse.ArgumentParser(description='Lab 1: Crawl Russian text pages')
    parser.add_argument('urls', nargs='+', help='Seed URLs to start crawling')
    parser.add_argument('--min-pages', type=int, default=100,
                        help='Minimum pages to collect (default: 100)')
    parser.add_argument('--min-words', type=int, default=1000,
                        help='Minimum words per page (default: 1000)')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between requests in seconds (default: 1.0)')
    parser.add_argument('--output', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Output directory (default: script directory)')
    args = parser.parse_args()
    crawl(args.urls, args.output, args.min_pages, args.min_words, args.delay)


if __name__ == '__main__':
    main()
