import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score


#     Lower -> Tokenize -> Lemmatize -> Remove stop words and non-alpha
#     Returns cleaned sentence
def clean_text(entry):
    #     Create tag map for use by the pos tagger
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    entry = entry.lower()
    entry = word_tokenize(entry)
    final = []
    lemmatizer = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_final = lemmatizer.lemmatize(word, tag_map[tag[0]])
            final.append(word_final)
    return str(final)


# Takes all rows from df['body'], cleans them and adds the to the new column df['text']
def preprocess(df):
    #     Clean body text and store all cleaned text in new df['text'] column
    text = []
    for index, entry in enumerate(df['body']):
        final = clean_text(entry)
        text.append(final)

    df['text'] = pd.Series(text)
    return df


# Loads labeled csv data into a pandas dataframe
def load_data(cleaned=False):
    if cleaned:
        return pd.read_csv('data/reddit_scraped.csv', encoding="ISO-8859-1")
    return preprocess(pd.read_csv('data/labeled_data.csv', encoding="ISO-8859-1"))


# Trains an SVM model on the df['text'] column, prints accuracy and returns the SVM model and tf idf vectorizer
def train(df):
    #     Separate into train/test
    train_x, test_x, train_y, test_y = model_selection.train_test_split(df['text'], df['class'], test_size=0.2)

    #     Encode class labels
    encoder = LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    test_y = encoder.fit_transform(test_y)

    #     Convert x to tf idf
    tfidf = TfidfVectorizer(max_features=4000)
    tfidf.fit(df['text'])
    train_x_tfidf = tfidf.transform(train_x)
    test_x_tfidf = tfidf.transform(test_x)

    #     Train SVM model
    SVM = svm.SVC(kernel='linear')
    SVM.fit(train_x_tfidf, train_y)

    #     Use model to predict on test set
    predict_SVM = SVM.predict(test_x_tfidf)

    #     Calculate accuracy
    print("SVM Accuracy Score -> ", accuracy_score(predict_SVM, test_y) * 100)
    return SVM, tfidf


# Takes a model, a tf idf vectorizer and a text string
# Returns predicted label (from original [-1,0,1] classification)
def classify(model, tfidf, text):
    text = clean_text(text)
    x = tfidf.transform([text])
    y = model.predict(x)
    return y - 1  # original labels are 1 off from those returned by the model


# ==== Example usage ====

# Load data -> Train SVM model
df = load_data(cleaned=True)
df = preprocess(df)
model, tfidf = train(df)
