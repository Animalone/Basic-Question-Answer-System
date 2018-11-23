#student name: Wuang Shen
#User name: wuangs
#Team name: Wuang Shen
#student id: 716090
import pickle
import nltk
import json
import time
from sklearn.metrics import f1_score
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import re
import spacy
import csv
from nltk.corpus import stopwords
from spacy.tokens import Doc
START = time.time()
# with open("testing.json") as f:
#     data = json.load(f)
with open("documents.json") as f:
    documents = json.load(f)
with open("testing.json") as f:
    data_devel = json.load(f)   
with open('test_prediction.pkl', 'rb') as f:
	devel_tags = pickle.load(f)
with open('predictSentenceTest.pkl', 'rb') as f:
	predicted_sentences = pickle.load(f)

stopwords = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')
probs = {w.orth: w.prob for w in nlp.vocab}
usually_titled = [w for w in nlp.vocab if w.is_title and probs.get(nlp.vocab[w.orth].lower, -10000) < probs.get(w.orth, -10000)]

for lex in usually_titled:
   lower = nlp.vocab[lex.lower]
   lower.shape = lex.shape
   lower.is_title = lex.is_title
   lower.cluster = lex.cluster
   lower.is_lower = lex.is_lower

def lemmaSentence(sentence):
	new_sentence = []
	for item in sentence:
		new_sentence.append(item.lemma_)
	return Doc(nlp.vocab, words = new_sentence)


def predict_Answer(sentence, question, tag):
	if tag == 'OTHER':
		result = None
		level = 0
		for chunk in sentence.noun_chunks:
			if not chunk.lemma_ in question.text:
				if level == 0:
					result = chunk.text
				if chunk.root.text in question.text and not chunk.root.text.lower() in stopwords:
					if level != 2:
						result = chunk.text
						level = 1
				if chunk.root.head.text in question.text and not chunk.root.head.text.lower() in stopwords:
					if level != 2:
						result = chunk.text
						level = 2
		if result:
			return result.lower()
		else:
			return "None_OTHER"
	else:
		answer = str()
		for token in sentence:
			if len(answer)> 0 and token.ent_type_ != tag:
				break
			if token.ent_type_ == tag:
				if not len(answer):
					answer += token.text.encode('ascii', 'ignore')+' '
				else:
					if previous == tag:
						answer += token.text.encode('ascii', 'ignore')+' '
			previous = token.ent_type_
		if len(answer):
			answer = answer[:len(answer)-1]
			answer = answer.lower()
			return answer.decode("utf-8")
		else:
			return "None"

predictList = []
index = 0
for item in data_devel:
	docid = item["docid"]
	question = item["question"]
	question = lemmaSentence(nlp(question))
	predicted_sentence =  predicted_sentences[index]
	tag = devel_tags[index]
	doc = nlp(predicted_sentence)
	answerP = predict_Answer(doc, question, tag)
	predictList.append(answerP.encode('ascii', 'ignore'))
	index += 1

END = time.time()
print(END-START)
with open('result.csv', 'w') as csvfile:
    fieldnames = ['id', 'answer']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(data_devel)):
    	row = {}
    	row['id'] = i
    	answer = predictList[i]
    	answer = answer.replace(' ,','')
    	answer = answer.replace(',','')
    	answer = answer.replace('"','')
    	row['answer'] = answer
    	writer.writerow(row)
# print "marco:",f1_score(answerList, predictList, average='macro')
# print "mirco:",f1_score(answerList, predictList, average='micro')
# print "weighted:",f1_score(answerList, predictList, average='weighted')