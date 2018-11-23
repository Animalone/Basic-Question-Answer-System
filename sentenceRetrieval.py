#student name: Wuang Shen
#User name: wuangs
#Team name: Wuang Shen
#student id: 716090
import json
import nltk
import re
import pickle
from nltk.tokenize import word_tokenize
from math import log
from sklearn.metrics import f1_score
from collections import defaultdict, Counter
from nltk.tag import StanfordNERTagger
import spacy
stopwords = set(nltk.corpus.stopwords.words('english')) # wrap in a set() (see below)

nlp = spacy.load('en_core_web_sm')
probs = {w.orth: w.prob for w in nlp.vocab}
usually_titled = [w for w in nlp.vocab if w.is_title and probs.get(nlp.vocab[w.orth].lower, -10000) < probs.get(w.orth, -10000)]

for lex in usually_titled:
   lower = nlp.vocab[lex.lower]
   lower.shape = lex.shape
   lower.is_title = lex.is_title
   lower.cluster = lex.cluster
   lower.is_lower = lex.is_lower

def extract_terms(doc):
    doc   = nlp(doc)
    # tokenize the sentence 
    tokens = []
    # remove all stop words 
    for token in doc:
        if not token.is_stop:
            tokens.append(token.lemma_)
    return tokens 

def get_term_frequencies(documents):
    tf = defaultdict(dict)
    total_docs  = len(documents)
    for doc in documents:
        doc_id = documents.index(doc)
        doc = extract_terms(doc)
        for term in doc:
            tf[term][doc_id] = tf[term].get(doc_id, 0) + 1 
    return tf, total_docs

def retrieve_sentence(sentence, query):
    scores = {}
    terms = extract_terms(query)
    for term in terms:
        if term in sentence:
            posting_list = sentence[term]
            for doc_id, weight in posting_list.items():
                scores[doc_id] = scores.get(doc_id, 0) + weight
    scores = [(k, v) for k, v in scores.items()]
    scores = sorted(scores, key=lambda t:t[1], reverse=True)
    return scores

def get_okapibm25(tf, total_docs, documents):
    k1, b, k3 = 1.5, 0.5, 0
    okapibm25 = defaultdict(dict)

    total = 0
    for d in documents:
        total += len(d)
    avg_doc_length = total/len(documents)*1.0

    for term, doc_list in tf.items():
        df = len(doc_list)
        for doc_id, freq in doc_list.items():
            qtf = 1.2
            idf = log((total_docs-df+0.5) / df+0.5)
            tf_Dt = ((k1+1)*tf[term][doc_id]) / (k1*((1-b)+b*(len(documents[doc_id])/avg_doc_length) + tf[term][doc_id]))
            if qtf == 0:
                third = 0
            else:
                third = ((k3+1)*qtf) / (k3+qtf)
                okapibm25[term][doc_id] = idf*tf_Dt*third
    return okapibm25

with open("testing.json") as f:
    data = json.load(f)
with open("documents.json") as f:
    documents = json.load(f)

docid_set = set()
for item in data:
    docid_set.add(item["docid"])
docid_list = list(docid_set)
docid_list.sort()


document_list = []
for item in documents[docid_list[0]:docid_list[-1]+1]:
    document_list.append(item)

docs = {}
for item in document_list:
    docid = item["docid"]
    text = item["text"]
    tf, total_docs = get_term_frequencies(text)
    okapibm25 = get_okapibm25(tf, total_docs, text)
    docs[docid] = okapibm25

predict = []

index = 0
count = 0

for item in data:
    question = item["question"]
    docid = item["docid"]
    try:
        predict.append(retrieve_sentence(docs[docid], question)[0][0])
    except:
        predict.append(0)

index = 0
paragraph = {}
for item in data:
    docid = item["docid"]
    sentences = documents[docid]["text"][predict[index]].split('. ')
    tf, total_docs = get_term_frequencies(sentences)
    okapibm25 = get_okapibm25(tf, total_docs, sentences)
    paragraph[index] = okapibm25
    index += 1

index = 0
aka = 0
predictSentence = []
for item in data:
    docid = item["docid"]
    question = item["question"]
    try:
        predicted_sentenceid = retrieve_sentence(paragraph[index], question)[0][0]
    except:
        predicted_sentenceid = 0
        aka += 1
    predictSentence.append(documents[docid]["text"][predict[index]].split('. ')[predicted_sentenceid])
    index += 1

outputfile = open('predictSentenceTest.pkl','wb')
pickle.dump(predictSentence,outputfile,protocol=0)
outputfile.close()
print("Run successfully")