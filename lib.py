
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn import model_selection, naive_bayes, svm
from sklearn.svm import SVC
import sklearn.svm._classes
import nltk
import pickle
import threading


class Processor:
    def __init__(self):
        self.downloadMissing()
        self.model = self.load('svm_model')
        self.tfidf = self.load('tfidf')

    def load(self, name):
        with open(name, 'rb') as f:
            item = pickle.load(f)
        return item

    def downloadMissing(self):
        try:
            nltk.data.find('wordnet')
        except LookupError:
            nltk.download('wordnet')
        try:
            nltk.data.find('punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
        try:
            nltk.data.find('stopwords')
        except LookupError:
            nltk.download('stopwords')

    def classify(self, sentence, callback):
        w = self.Worker(sentence, callback, self.model, self.tfidf)
        w.start()

    class Worker(threading.Thread):
        def __init__(self, sentence, callback, model, tfidf):
            threading.Thread.__init__(self)
            self.callback = callback
            self.sentence = sentence
            self.model = model
            self.tfidf = tfidf

        def run(self):
            label = self.classify(self.model, self.tfidf, self.sentence)
            self.callback((self.sentence, label))

        def clean_text(self, entry):
            tag_map = defaultdict(lambda: wn.Noun)
            tag_map['J'] = wn.ADJ
            tag_map['V'] = wn.VERB
            tag_map['R'] = wn.ADV

            # lower -> tokenize -> lemmatize -> remove stop words and non-alpha -> return cleaned sentences
            entry = entry.lower()
            entry = word_tokenize(entry)
            final = []
            lemmatizer = WordNetLemmatizer()
            for word, tag in pos_tag(entry):
                if word not in stopwords.words('english') and word.isalpha():
                    word_final = lemmatizer.lemmatize(word, tag_map[tag[0]])
                    final.append(word_final)
            return str(final)

        def preprocess(self, df):
            text = []
            for index, entry in enumerate(df['body']):
                final = self.clean_text(entry)
                text.append(final)

        def load_data(self, cleaned=False):
            if cleaned:
                return pd.read_csv('data/cleaned_data.csv', encoding='ISO-8859-1')
            return self.preprocess(pd.read_csv('data/reddit_scraped.csv', encoding='ISO-8859-1'))

        def train(self, df, print_accuracy=False):
            train_x, test_x, train_y, test_y = model_selection.train_test_split(df['text'], df['class'], test_size=0.2)

            encoder = LabelEncoder()
            train_y = encoder.fit_transform(train_y)
            test_y = encoder.fit_transform(test_y)

            tfidf = TfidfVectorizer(max_features=4000)
            tfidf.fit(df['text'])
            train_x_tfidf = tfidf.transform(train_x)
            test_x_tfidf = tfidf.transform(test_x)

            SVM = SVC(kernel='linear')
            SVM.fit(train_x_tfidf, train_y)

            if print_accuracy:
                predict_SVM = SVM.predict(test_x_tfidf)
                print("SVM Accuracy Score -> ", accuracy_score(predict_SVM, test_y) * 100)

            return SVM, tfidf

        def classify(self, model, tfidf, text):
            text = self.clean_text(text)
            x = tfidf.transform([text])
            y = model.predict(x)
            return y - 1

        def save(self, item, name):
            with open(name, 'wb') as f:
                pickle.dump(item, f)



















