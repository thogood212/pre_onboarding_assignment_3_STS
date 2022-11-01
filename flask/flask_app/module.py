from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import re

class MODEL():
    def __init__(self, model_path, model_name):
        # load checkpoint file
        # Load trained model
        self.model_path = model_path
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)
        #load tokenizer
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def forward(self, sentence1, sentence2):
        tokens = self.tokenizer(sentence1, sentence2,padding=True, truncation=True, return_tensors="pt")
        output = self.model(**tokens)
        return output

def pre_treat(sent):
    sentence = sent.strip()
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
    sentence = hangul.sub('', sentence)
    return sentence