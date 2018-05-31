
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


# --------------------------------------------------------------------------------------
# import pdfs and extract data from them, in appropriate format

# good CV data
# cv_dir = '/home/laura/cv-project/Business_CV_Books/BK1'
cv_dir = '/Users/rukayajohaadien/Dropbox (Prodigy)/The Deloreans/Careers Data (CVs)/Good CVs'
cv_list = glob.glob('{}/*.pdf'.format(cv_dir))
# hack for the moment -- repeat CV book pdfs four times,
#  to have comparable number of CVs in both categories
#  (do this properly in a real model!)
cv_list = cv_list + cv_list + cv_list + cv_list
pdf_names_cvbook, pdf_text_cvbook, stem_dict_0 = get_pdf_text(cv_list)
labels_cvbook = np.ones(len(pdf_text_cvbook), dtype=np.int)

# bad CV data
#cv_dir = '/home/laura/cv-project/community_collected_CVs'
cv_dir = '/Users/rukayajohaadien/Dropbox (Prodigy)/The Deloreans/Careers Data (CVs)/Bad CVs'
cv_list = glob.glob('{}/*.pdf'.format(cv_dir))
pdf_names_community, pdf_text_community, stem_dict_1 = get_pdf_text(cv_list)
labels_community = np.zeros(len(pdf_text_community), dtype=np.int)

# combine for processing
pdf_text = pdf_text_cvbook + pdf_text_community
stem_dict = {**stem_dict_0, **stem_dict_1}
labels = np.concatenate([labels_cvbook, labels_community])

# --------------------------------------------------------------------------------------
# determine word vectors from pdf corpus

print('\n-------------------------------------------')
print('Building doc2vec model\n')

# vector_size = 200
# model = doc2vec.Doc2Vec(
#     min_count=5, # ignore words with word counts less than this
#     window=10,
#     size=vector_size,
#     sample=1e-4,
#     negative=5,
#     workers=2,
#     max_vocab_size=10000,
#     dm=1,
# )
# model.build_vocab(pdf_text)
# print('Vocab size:', len(model.wv.vocab))

# # train the word vector model (shuffle sentences each time)
# for epoch in range(20):
#     print(epoch)
#     shuffle(pdf_text)
#     model.train(
#         pdf_text,
#         total_examples=model.corpus_count,
#         epochs=model.iter
#     )

gn = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(gn, binary=True)

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

word_to_check = 'languages'
print(word_to_check)
p_stemmer = PorterStemmer()
check_word = p_stemmer.stem(word_to_check)
print(model.wv.most_similar(check_word))

# --------------------------------------------------------------------------------------

cv_vectors = [model.infer_vector(doc.words) for doc in pdf_text]

# --------------------------------------------------------------------------------------
# get features for fitting

train_percentage = 0.7
train_x, test_x, train_y, test_y = train_test_split(
    cv_vectors, labels, train_size=train_percentage)

# --------------------------------------------------------------------------------------
# Logistic regression

print('\n-------------------------------------------')
print('Fit results:')

regression = LogisticRegression()
regression.fit(train_x, train_y)
lr_score = regression.score(test_x, test_y)
print()
print('Logistic regression accuracy:', lr_score)

# --------------------------------------------------------------------------------------
# Random forest

# fit
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_x, train_y)

# check fit results
result = forest.predict(test_x)
diff = result == test_y
accuracy = diff.astype(int).sum() / len(diff)
print()
print('Random forest accuracy:', accuracy)

# --------------------------------------------------------------------------------------
# investigate some results

feature_importances = forest.feature_importances_
sort_indices = np.argsort(feature_importances)[::-1]

print('\n-------------------------------------------')
print('Most important rendom forest vector features')
print('  and representative similar words along the vector axis (positive and negative)')
for ii in sort_indices[0:6]:
    print()
    print(ii, feature_importances[ii])
    vec = np.zeros(vector_size)
    vec[ii] = 1
    print(model.wv.similar_by_vector(vec))
    print(model.wv.similar_by_vector(-vec))

# visualise words that appear the most
vocab = list(model.wv.vocab.keys())
word_counts = [model.wv.vocab[w].count for w in vocab]
word_order = np.argsort(word_counts)[::-1]
top_words = [vocab[ii] for ii in word_order[0:200]]
plot_words(model, top_words)

# check full word for stem:
print('\n-------------------------------------------')
print('Check some stems:')

stemmed_word = 'haa'
possible_full_words = [w for w in stem_dict.keys() if w.startswith(stemmed_word)]
print('{}: {}'.format(stemmed_word, possible_full_words))

stemmed_word = 'econom'
possible_full_words = [w for w in stem_dict.keys() if w.startswith(stemmed_word)]
print('{}: {}'.format(stemmed_word, possible_full_words))
