#student name: Wuang Shen
#User name: wuangs
#Team name: Wuang Shen
#student id: 716090
import spacy
import json
import pickle
from spacy.attrs import ORTH, LEMMA, TAG, ENT_TYPE, ENT_IOB

with open("training.json") as f:
	data_training = json.load(f)
with open("devel.json") as f:
	data_devel = json.load(f)

nlp = spacy.load('en_core_web_sm')
probs = {w.orth: w.prob for w in nlp.vocab}
usually_titled = [w for w in nlp.vocab if w.is_title and probs.get(nlp.vocab[w.orth].lower, -10000) < probs.get(w.orth, -10000)]

for lex in usually_titled:
   lower = nlp.vocab[lex.lower]
   lower.shape = lex.shape
   lower.is_title = lex.is_title
   lower.cluster = lex.cluster
   lower.is_lower = lex.is_lower

training = []
count = 0
for item in data_training:
	label = 0
	doc = nlp(item["text"])
	for ent in doc.ents:
		training.append(ent.label_)
		label = 1
		break
	if not label:
		training.append('OTHER')
	print count 
	count += 1
    


outputfile = open('training.pkl','wb')
pickle.dump(training,outputfile,protocol=0)
outputfile.close()