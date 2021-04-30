import requests

def count_words_at_url(url):
    resp = requests.get(url)
    words = len(resp.text.split())
    print(f'{words} words in {url}')
    return words
