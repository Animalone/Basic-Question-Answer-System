#student name: Wuang Shen
#User name: wuangs
#Team name: Wuang Shen
#student id: 716090
import pickle
import json
import nltk
import spacy
import time
from nltk.tag import StanfordNERTagger
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation 
from sklearn.metrics import accuracy_score, classification_report

START = time.time()
nlp = spacy.load('en_core_web_sm')
probs = {w.orth: w.prob for w in nlp.vocab}
usually_titled = [w for w in nlp.vocab if w.is_title and probs.get(nlp.vocab[w.orth].lower, -10000) < probs.get(w.orth, -10000)]

for lex in usually_titled:
   lower = nlp.vocab[lex.lower]
   lower.shape = lex.shape
   lower.is_title = lex.is_title
   lower.cluster = lex.cluster
   lower.is_lower = lex.is_lower

with open('training.pkl', 'rb') as f:
	answer_tags = pickle.load(f)

with open("training.json") as f:
	data = json.load(f)

with open("testing.json") as f:
    test_data = json.load(f)

def get_BOW(doc):
    BOW = {}
    for token in doc:
        BOW[token.lemma_] = BOW.get(token.lemma_,0) + 1
    return BOW

def prepare_data(feature_extractor, tags, dataset, vectorizer, mode):
    feature_matrix = []
    classifications = []
    index = 0
    for item in dataset:
        doc = nlp(item["question"])
        feature_dict = feature_extractor(doc)   
        feature_matrix.append(feature_dict)
        classifications.append(tags[index])
        index += 1
     
    if mode == 'train':
    	dataset = vectorizer.fit_transform(feature_matrix)
    if mode == 'devel':
    	dataset = vectorizer.transform(feature_matrix)
    return dataset,classifications

def prepare_test_data(feature_extractor, dataset, vectorizer):
    feature_matrix = []
    for item in dataset:
        doc = nlp(item["question"])
        feature_dict = feature_extractor(doc)   
        feature_matrix.append(feature_dict)
    dataset = vectorizer.transform(feature_matrix)
    return dataset

def check_results(predictions, classifications):
    print("accuracy")
    print(accuracy_score(classifications,predictions))
    print(classification_report(classifications,predictions))

vectorizer = DictVectorizer()
dataset,classifications = prepare_data(get_BOW, answer_tags, data, vectorizer, mode = 'train') 
dataset_test= prepare_test_data(get_BOW, test_data, vectorizer)

clf = RandomForestClassifier(n_estimators= 50)
clf.fit(dataset,classifications)
predictions = clf.predict(dataset_test)
# check_results(predictions, classifications_devel)
END = time.time()
print(END-START)

outputfile = open('test_prediction.pkl','wb')
pickle.dump(predictions,outputfile,protocol=0)
outputfile.close()