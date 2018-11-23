#student name: Wuang Shen
#User name: wuangs
#Team name: Wuang Shen
#student id: 716090s

The package includes 5 files- tagTrainingQuestion.py, classify.py,
sentenceRetrieval.py, predictTest.py and report.pdf

tagTrainingQuestion.py - this script using spaCy NER feature to tag each question in training dataset into pre-defined categories. And it will output a pickle file.

classify.py - this script builds random forest classifer on training dataset, and classify question in testing dataset to pre-defined categories. And it will output a pickle file. 

sentenceRetrieval.py - this script use bm25 methods retrieval relevant sentence from document file. And it will output a pickle file.

predictTest.py- this script will extract answers from retrieved sentence, based on question's answer type, and sentence dependency. And it will output a csv file. 

report.pdf - this file is the final report for this project. 
------------------------------------------------
The execution of those script should be followed by orders below:
python tagTrainingQuestion.py
python classify.py
python sentenceRetrieval.py
python predictTest.py
------------------------------------------------

Total time required to run those scripts will be within 1~2 hours. 