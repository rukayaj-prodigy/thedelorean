
import PyPDF2
import glob
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import re

stops = set(stopwords.words("english"))


def preprocess_words(pdf_content):

    # take out numbers (for now)
    letters_only = re.sub("[^a-zA-Z]", " ", pdf_content)

    word_list = letters_only.lower().split()

    # take out one and two character words
    meaningful_words = [w for w in word_list if len(w) > 2]
    # take out stop words
    meaningful_words = [w for w in meaningful_words if w not in stops]
    return meaningful_words


def get_pdf_text(cv_list):
    pdf_names = []
    pdf_text = []
    for cv_file in sorted(cv_list):
        f = open(cv_file, 'rb')

        filename_only = Path(cv_file).name
        cv = PyPDF2.PdfFileReader(f)
        n_pages = cv.getNumPages()
        pdf_content = ''
        for n in range(n_pages):
            page = cv.getPage(n)
            page_content = page.extractText()
            pdf_content += page_content

        meaningful_word_list = preprocess_words(pdf_content)

        pdf_names.append(filename_only)
        pdf_text.append(' '.join(meaningful_word_list))

    return pdf_names, pdf_text


def get_vocab_features(pdf_text):
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=5000,
                                 min_df=10)

    features = vectorizer.fit_transform(pdf_text)
    features = features.toarray()
    vocab = vectorizer.get_feature_names()

    # Sum up the counts of each vocabulary word
    vocab_counts = np.sum(features, axis=0)

    return vocab, vocab_counts, features


# --------------------------------------------------------------------------------------
# preprocess to get vocab

# good CV data
cv_dir = '/home/laura/cv-project/Business_CV_Books/BK1'
cv_list = glob.glob('{}/BK*-CV*.pdf'.format(cv_dir))
pdf_names_cvbook, pdf_text_cvbook = get_pdf_text(cv_list)
labels_cvbook = np.ones_like(pdf_text_cvbook, dtype=np.int)

# bad CV data
cv_dir = '/home/laura/cv-project/community_collected_CVs'
cv_list = glob.glob('{}/*.pdf'.format(cv_dir))
pdf_names_community, pdf_text_community = get_pdf_text(cv_list)
labels_community = np.zeros_like(pdf_text_community, dtype=np.int)

pdf_text = pdf_text_cvbook + pdf_text_community
labels = np.concatenate([labels_cvbook, labels_community])

# --------------------------------------------------------------------------------------

vocab, vocab_counts, features = get_vocab_features(pdf_text)
counts_total = np.sum(vocab_counts)

df_words_summary = pd.DataFrame({
    'vocab': vocab,
    'counts': vocab_counts,
    'counts_norm': vocab_counts / counts_total
})
print(df_words_summary.sort_values('counts_norm', ascending=False))

# --------------------------------------------------------------------------------------
# get features for fitting

train_percentage = 0.7
train_x, test_x, train_y, test_y = train_test_split(
    features, labels, train_size=train_percentage)

# --------------------------------------------------------------------------------------
# fit
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_x, train_y)

# check fit results
result = forest.predict(test_x)
diff = result == test_y
accuracy = diff.astype(int).sum() / len(diff)
print('Random forest accuracy:', accuracy)

# --------------------------------------------------------------------------------------
# find feature importance
#   - this currently shows that this fitting is a bit rubbish because a lot of the features are
#     nonsense

feature_importances = forest.feature_importances_
df_features = pd.DataFrame({
    'feature': vocab,
    'importance': feature_importances
})
sorted_features = df_features.sort_values('importance', ascending=False)

sorted_features[0:40].plot(x='feature', y='importance', kind='bar', fontsize=16)
plt.show()
