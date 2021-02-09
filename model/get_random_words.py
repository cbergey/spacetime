from __future__ import absolute_import, division, print_function, unicode_literals
import io
import os
from gensim import utils
import gensim.models
import gensim.models.word2vec
from gensim.test.utils import datapath             
import numpy as np   
import csv

model = gensim.models.Word2Vec.load("trained_models/childes_adult_word2vec.model")

max = len(model.wv.vocab) - 1
words = []
for i in range(0, 3000):
    thisword = model.wv.index2word[np.random.randint(0, max)]
    words.append(thisword)
    print(thisword)

with open('childes_random_words.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["word"])
    for word in words:
        writer.writerow([word])