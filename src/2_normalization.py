import numpy as np
import torch
import torchtext
from torchtext import data
import spacy
import nltk
import random

SEED = 1234

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
ps = nltk.stem.porter.PorterStemmer()

# Text Normalization


def tokenizer (sentence):
    tk = nltk.word_tokenize(sentence)
    for i in range(len(tk)):
        tk[i] = ps.stem(tk[i]) # Stemming
    return tk

def text_normalization():
	TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer, lower=True)
	LABEL = torchtext.data.LabelField(sequential=False, dtype=torch.float)
	fields = [(None, None), (None, None), ('label', LABEL), (None, None), ('text', TEXT)]

	# Generate dataset for torchtext
	train_data, test_data = torchtext.data.TabularDataset.splits(path='../data/input', 
		train='train', test='test', format = 'tsv', fields=fields)

	train_data, val_data = train_data.split(random_state=random.seed(SEED))

	# Build vocab
	TEXT.build_vocab(train_data, vectors="glove.6B.100d") #
	LABEL.build_vocab(train_data)

	# Create batch and iterate dataset
	BATCH_SIZE = [64, 64, 64]
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	train_iter, val_iter, test_iter = torchtext.data.Iterator.splits(
	        (train_data, val_data, test_data),
	        batch_sizes=BATCH_SIZE, device=device)
