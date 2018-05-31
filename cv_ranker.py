import glob
from pathlib import Path
import PyPDF2
import re

import numpy as np
import matplotlib.pylab as plt
from random import shuffle

from gensim.models import doc2vec, KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import decomposition

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
import pdb
from load_cvs import get_pdf_text, is_proper_cv
import helpers

# Set up natural language toolkit
nltk.download('stopwords')
stops = set(stopwords.words("english"))

# Load training data
cv_dir = '/Users/rukayajohaadien/Dropbox (Prodigy)/The Deloreans/Careers Data (CVs)'
good_cv_list = glob.glob('{}/Good CVs/*.pdf'.format(cv_dir))
bad_cv_list = glob.glob('{}/Bad CVs/*.pdf'.format(cv_dir))

good_names, good_text, good_dict = helpers.get_pdf_text(good_cv_list)
good_labels = np.ones(len(good_text), dtype=np.int)
bad_names, bad_text, bad_dict = helpers.get_pdf_text(good_cv_list)
bad_labels = np.ones(len(good_text), dtype=np.int)

# Combine for processing
text = good_text + bad_text
stem_dict = {**good_dict, **bad_dict}
labels = np.concatenate([good_labels, bad_labels])

# Build vectors
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# --------------------------------------------------------------------------------------
# some sanity checks -- look for most similar words, to see all makes sense

print('\n-------------------------------------------')
print('Check for some word similarities in our vector model:')

print()
word_to_check = 'management'
print(word_to_check)
p_stemmer = PorterStemmer()
check_word = p_stemmer.stem(word_to_check)
print(model.wv.most_similar(check_word))
