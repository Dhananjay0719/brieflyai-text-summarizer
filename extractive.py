import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

nlp = spacy.load("en_core_web_sm")

def extractive_summary(text, ratio=0.3):
    doc = nlp(text)
    stopwords = STOP_WORDS

    word_freq = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text not in punctuation:
            word_freq[word.text] = word_freq.get(word.text, 0) + 1

    max_freq = max(word_freq.values())
    for word in word_freq:
        word_freq[word] /= max_freq

    sent_scores = {}
    for sent in doc.sents:
        for word in sent:
            if word.text in word_freq:
                sent_scores[sent] = sent_scores.get(sent, 0) + word_freq[word.text]

    select_len = max(1, int(len(list(doc.sents)) * ratio))
    summary = nlargest(select_len, sent_scores, key=sent_scores.get)

    return " ".join([s.text for s in summary])