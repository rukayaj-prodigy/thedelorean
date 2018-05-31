
from gensim.models.doc2vec import LabeledSentence
from gensim.models import doc2vec, KeyedVectors

import glob
from pathlib import Path
import PyPDF2
import re

import numpy as np
import matplotlib.pylab as plt
from random import shuffle

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import decomposition

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
from load_cvs import is_proper_cv
nltk.download('stopwords')
stops = set(stopwords.words("english"))

def get_pdf_text(cv_list):
    """Extracts text from list of pdf files."""
    pdf_names, pdf_document_tagged = [], []
    stem_dict = {}
    for ii, cv_file in enumerate(sorted(cv_list)):
        f = open(cv_file, 'rb')

        filename_only = Path(cv_file).name
        cv = PyPDF2.PdfFileReader(f)
        n_pages = cv.getNumPages()
        pdf_content = ''
        for n in range(n_pages):
            page = cv.getPage(n)
            page_content = page.extractText()
            pdf_content += page_content

        if not is_proper_cv(page_content):
            print("Throwing away cv")
            print(page_content)
            continue

        tagged_document, stem_dict_doc = preprocess_lines(
            pdf_content,
            '{}-{}'.format(ii, filename_only),
        )
        stem_dict.update(stem_dict_doc)
        if tagged_document is not None:
            pdf_document_tagged.append(tagged_document)
            pdf_names.append(filename_only)

    return pdf_names, pdf_document_tagged, stem_dict


def stem_words(words):
    """Takes list of words and returns stems of those words,
    and a dictionary mapping modified stemmed words to their original word."""
    p_stemmer = PorterStemmer()
    stem_dict = {}
    stemmed_words = []

    for w in words:
        w_stem = p_stemmer.stem(w)
        stemmed_words.append(w_stem)
        if w_stem != w:
            stem_dict[w] = w_stem

    return stemmed_words, stem_dict


def preprocess_lines(pdf_content, document_tag):
    """Preprocesses a string of words. For the moment, the step are:
        - removes everything but letters
        - splits into lines
        - converts to lower case
        - splits line into words
        - takes stems of words
        - takes out one and two caracter words
        - takes out stop words (provided by nltk)
    """

    # take out numbers (for now)
    letters_only = re.sub("[^a-zA-Z\n]", " ", pdf_content)
    line_list = letters_only.split('\n')

    sentence_list = []
    stem_dict = {}
    for line in line_list:
        words = line.lower().split()
        # stem words
        meaningful_words, stem_dict_line = stem_words(words)
        stem_dict.update(stem_dict_line)
        # take out one and two character words
        meaningful_words = [w for w in meaningful_words if len(w) > 2]
        # take out stop words
        meaningful_words = [w for w in meaningful_words if w not in stops]

        if len(meaningful_words) != 0:
            sentence_list.extend(meaningful_words)

    if sentence_list != []:
        tagged_document = doc2vec.TaggedDocument(sentence_list, tags=[document_tag])
        return tagged_document, stem_dict
    else:
        return None, stem_dict


def plot_words(model, words):
    words_vectors = []
    for word in words:
        words_vectors.append(model.wv[word])

    # transform word vectors to 2D using PCA
    pca = decomposition.PCA(n_components=2)
    pca.fit(words_vectors)
    reduced = pca.transform(words_vectors)

    for word, vec in zip(words, reduced):
        x, y = vec[0], vec[1]
        plt.plot(x, y, 'k.')
        plt.annotate(word, xy=(x, y))
    plt.show()