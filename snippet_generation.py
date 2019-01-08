# Snippet Generation Techniques
# Vikram Sunil Bajaj (vsb259)
import serpscrap
from bs4 import BeautifulSoup
import requests
import re
from gensim.models import KeyedVectors
from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from nltk.corpus import wordnet


def get_google_links_snippets(query):
    """ retrieves top 10 results (for which snippets could be retrieved) along with URLs and snippets """
    config = serpscrap.Config()
    config.set('scrape_urls', False)
    config.set('do_caching', False)

    scrap = serpscrap.SerpScrap()
    scrap.init(config=config.get(), keywords=[query])
    results = scrap.scrap_serps()
    i = 0

    urls = []
    google_snippets = []

    for result in results:
        if result['serp_snippet'] and i < 10:
            urls.append(result['serp_url'])
            google_snippets.append(re.sub(r'[^\x00-\x7F]+', ' ', result['serp_snippet']).replace('\n', ''))
            i += 1

    return urls, google_snippets


# similarity estimation function - word mover's distance
print("Loading Google News Vectors ...")
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)  # uncomment, run once
model.init_sims(replace=True)  # uncomment and run once

# install gensim, and pyemd (need pyemd whl file from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyemd for Windows)
# Google News pre-trained model from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
# ref: https://markroxor.github.io/gensim/static/notebooks/WMD_tutorial.html


def get_wmd_dist(s1, s2, model):
    """ gets word mover's distance between sentences s1, s2 """
    s1 = s1.lower().strip().split()
    s2 = s2.lower().strip().split()

    distance = model.wmdistance(s1, s2)
    return distance


def display_wmd(google_snippets, my_snippets):
    """ function to print word mover's distance between Google's snippets and my snippets """
    for i in range(len(google_snippets)):
        print(i+1, '.', urls[i])
        print('Google Snippet:', google_snippets[i])
        print('Generated Snippet:', my_snippets[i])
        print("Word Mover's Distance:", get_wmd_dist(google_snippets[i], my_snippets[i], model))
        print()


def approach_1():
    print("QUERY-INDEPENDENT APPROACHES")
    print("1. Extracting Part of the Page")
    print("In this approach, the snippet is either the first 20 words of the meta description (if available) OR the "
          "first 20 words of the first paragraph that has at least 5 words. "
          "If no paragraphs are found, it uses the first 20 words of the page text.")
    my_snippets_1 = []
    for u in urls:
        snippet = ''
        res = requests.get(u)  # to do: handle failed requests/403s
        soup = BeautifulSoup(res.content, 'lxml')
        # check if page has meta description
        desc = soup.find_all('meta', {'name': 'description'})
        if desc:
            snippet += ' '.join(desc[0].get("content").split()[:20]) + ' ...'
            my_snippets_1.append(snippet)
        else:  # no meta description
            # attempt to grab paragraph text
            paragraphs = [re.sub(r'[^\x00-\x7F]+',' ', x.text).replace('\n', '').strip() for x in soup.find_all('p')]
            paragraphs = [p for p in paragraphs if len(p.split())>=5]  # paragraphs that have at least k words
            if len(paragraphs) != 0:
                # get first k words of first paragraph as snippet
                snippet += ' '.join(paragraphs[0].split()[:20]) + ' ...'
                my_snippets_1.append(snippet)
            else:
                # no paragraphs were extracted, use body text
                snippet += ' '.join(soup.find('body').text.replace('\n','').replace('\r','').strip().split()[:20]) + ' ...'
                my_snippets_1.append(snippet)

    print('Approach 1 Snippets:')
    display_wmd(google_snippets, my_snippets_1)


def approach_2():
    print()
    print("2. Page Summarization")
    print(
        "This approach uses TextRankSummarizer from sumy to summarize the page. It then extracts the first 20 words of "
        "the summary. Note that even gensim's summarize() method uses TextRank, but needs more work since it doesn't "
        "automatically get text from the URL. So, I used sumy instead, which uses an in-built HTMLParser to extract "
        "page text. ")
    my_snippets_2 = []
    summarizer = TextRankSummarizer()
    for u in urls:
        parser = HtmlParser.from_url(u, Tokenizer("english"))
        summary = summarizer(parser.document, 5)  # 5-line summary
        summary_str = ' '.join([str(sentence) for sentence in summary])
        snippet = ' '.join(summary_str.split()[:20]) + ' ...'
        my_snippets_2.append(snippet)

    display_wmd(google_snippets, my_snippets_2)


def approach_3():
    print()
    print("QUERY-DEPENDENT APPROACHES")
    print("3. Extracting Parts of the Page Having Query Terms")
    print("This approach first extracts page text, then checks if any sentences contain all query terms. Those "
          "sentences are shown first. Then, sentences that contain at least 1 query term are shown.")
    my_snippets_3 = []
    for u in urls:
        snippet_sentences = []
        parser = HtmlParser.from_url(u, Tokenizer("english"))
        res = requests.get(u)  # to do: handle failed requests/403s
        soup = BeautifulSoup(res.content, 'lxml')
        # check if page has meta description
        desc = soup.find_all('meta', {'name': 'description'})
        desc_str = ''
        if desc:
            desc_str = desc[0].get("content")

        combined_sentences = [str(s) for s in parser.document.sentences] + [desc_str]
        for s in combined_sentences:
            if all([q.lower() in str(s).lower().split() for q in
                    query.split()]):  # check if any sentence has all query terms
                snippet_sentences.append(str(s))
            if any([q.lower() in str(s).lower().split() for q in
                    query.split()]):  # check for sentences that have at least 1 query term
                if str(s) not in snippet_sentences:  # avoid duplicates
                    snippet_sentences.append(str(s))
        snippet_sentences_combined = ' '.join([s for s in snippet_sentences])
        snippet = ' '.join(snippet_sentences_combined.split()[:20]) + ' ...'
        my_snippets_3.append(snippet)

    display_wmd(google_snippets, my_snippets_3)


def approach_4():
    print()
    print("4. Extracting Parts of the Page Having Query Terms and Synonyms")
    print("This approach is similar to the previous one but also considers synonyms of query terms, "
          "i.e. it also looks for sentences that contain at least 1 query term synonym.")
    query_terms = query.split()
    synonyms_list = []

    # get synonyms of query terms using WordNet
    for q in query_terms:
        for syn in wordnet.synsets(q):
            for l in syn.lemmas():
                synonyms_list.append(l.name().replace('_', ' '))

    my_snippets_4 = []

    for u in urls:
        snippet_sentences = []
        parser = HtmlParser.from_url(u, Tokenizer("english"))
        res = requests.get(u)  # to do: handle failed requests/403s
        soup = BeautifulSoup(res.content, 'lxml')
        # check if page has meta description
        desc = soup.find_all('meta', {'name': 'description'})
        desc_str = ''
        if desc:
            desc_str = desc[0].get("content")
        combined_sentences = [str(s) for s in parser.document.sentences] + [desc_str]
        for s in combined_sentences:
            if all([q.lower() in str(s).lower().split() for q in
                    query.split()]):  # check if any sentence has all query terms
                snippet_sentences.append(str(s))
            if any([q.lower() in str(s).lower().split() for q in
                    query.split()]):  # check for sentences that have at least 1 query term
                if str(s) not in snippet_sentences:  # avoid duplicates
                    snippet_sentences.append(str(s))
            if any([syn.lower() in str(s).lower().split() for syn in
                    synonyms_list]):  # check for sentences that have at least one synonym
                if str(s) not in snippet_sentences:
                    snippet_sentences.append(str(s))
        snippet_sentences_combined = ' '.join([s for s in snippet_sentences])
        snippet = ' '.join(snippet_sentences_combined.split()[:20]) + ' ...'
        my_snippets_4.append(snippet)

    display_wmd(google_snippets, my_snippets_4)


def approach_5():
    print()
    print("5. Query-Based Summary")
    print("This approach gets the page summary using sumy and then extracts sentences from the summary "
          "that contain query terms or their synonyms.")
    query_terms = query.split()
    synonyms_list = []

    # get synonyms of query terms using WordNet
    for q in query_terms:
        for syn in wordnet.synsets(q):
            for l in syn.lemmas():
                synonyms_list.append(l.name().replace('_', ' '))

    my_snippets_5 = []
    summarizer = TextRankSummarizer()
    for u in urls:
        parser = HtmlParser.from_url(u, Tokenizer("english"))
        summary = summarizer(parser.document, 5)  # 5-line summary
        snippet_sentences = []
        for s in summary:
            # check if any sentence in summary has all query terms
            if all([q.lower() in str(s).lower().split() for q in query.split()]):
                snippet_sentences.append(str(s))
            # check for sentences that have at least 1 query term
            if any([q.lower() in str(s).lower().split() for q in query.split()]):
                if str(s) not in snippet_sentences:  # avoid duplicates
                    snippet_sentences.append(str(s))
            # check for sentences that have at least one synonym
            if any([syn.lower() in str(s).lower().split() for syn in synonyms_list]):
                if str(s) not in snippet_sentences:
                    snippet_sentences.append(str(s))
        snippet_sentences_combined = ' '.join([s for s in snippet_sentences])
        snippet = ' '.join(snippet_sentences_combined.split()[:20]) + ' ...'
        my_snippets_5.append(snippet)

    display_wmd(google_snippets, my_snippets_5)


if __name__ == "__main__":
    query = input("Enter Query: ").strip()
    urls, google_snippets = get_google_links_snippets(query)
    approach_1()
    approach_2()
    approach_3()
    approach_4()
    approach_5()
