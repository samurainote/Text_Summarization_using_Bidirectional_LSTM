
from numpy.random import seed
seed(1)

import gensim as gs
import pandas as pd
import numpy as np
import scipy as sc
import nltk
from nltk.tokenize import word_tokenize as wt
from nltk.tokenize import sent_tokenize as st
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.sequence import pad_sequences
import logging
import re
from collections import Counter

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# ========================================================================
#
# ========================================================================

# emb_size_all = 300
# maxcorp=5000
EMBEDDING_DIM = 300
VOCAB_SIZE = 5000


def createCorpus(t):
    corpus = []
    all_sent = []
    for k in t:
        for p in t[k]:
            corpus.append(st(p))
    for sent in range(len(corpus)):
        for k in corpus[sent]:
            all_sent.append(k)
    for m in range(len(all_sent)):
        all_sent[m] = wt(all_sent[m])

    all_words=[]
    for sent in all_sent:
        hold=[]
        for word in sent:
            hold.append(word.lower())
        all_words.append(hold)
    return all_words


def word2vecmodel(corpus):
    emb_size = emb_size_all
    model_type={"skip_gram":1,"CBOW":0}
    window=10
    workers=4
    min_count=4
    batch_words=20
    epochs=25
    #include bigrams
    #bigramer = gs.models.Phrases(corpus)

    model=gs.models.Word2Vec(corpus,size=emb_size,sg=model_type["skip_gram"],
                             compute_loss=True,window=window,min_count=min_count,workers=workers,
                             batch_words=batch_words)

    model.train(corpus,total_examples=len(corpus),epochs=epochs)
    model.save("%sWord2vec"%modelLocation)
    print('\007')
    return model










"""
generate summary length for test
"""

#train_data["nums_summ"]=list(map(lambda x:0 if len(x)<5000 else 1,data["articles"]))
#train_data["nums_summ"]=list(map(len,data["summaries"]))
#train_data["nums_summ_norm"]=(np.array(train_data["nums_summ"])-min(train_data["nums_summ"]))/(max(train_data["nums_summ"])-min(train_data["nums_summ"]))
