import glob
import numpy
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.svm import OneClassSVM

from joblib import dump, load

from util import file_get_contents

corpus = []
result_dir = '/home/leonardo/Documents/Workspace/Quali/code/leo-cnn-pp/data'

dataset_result_train_dir = result_dir + '/train'
dataset_result_test_dir = result_dir + '/test'


dataset_result_train_positive_dir = dataset_result_train_dir + '/' + "Negative" + '/'
dataset_result_test_positive_dir = dataset_result_test_dir + '/' + "Negative" + '/'

# TRAINING
positive_conversations = glob.glob(os.path.join(dataset_result_train_positive_dir, '*.txt'))
for positive_conversation in positive_conversations:
    file_content = file_get_contents(positive_conversation).lower().replace('\n', ' ').replace('\r', '')
    corpus.append(file_content)

vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 1))
X_train = vectorizer.fit_transform(corpus)
# print(vectorizer.get_feature_names())
# print(X)

for nu_value in numpy.arange(0.100, 0.999, 0.001):

    print("nu_value =", float(format(round(nu_value, 3))))

    oneclass = OneClassSVM(kernel='linear', nu=float(format(round(nu_value, 3))))
    oneclass.fit(X_train)

    y_train = oneclass.predict(X_train)
    # print(y_train)
    print(y_train[y_train == 1].size / y_train.size)

    # TESTING

    corpus = []

    test_positive_conversations = glob.glob(os.path.join(dataset_result_test_positive_dir, '*.txt'))
    for positive_conversation in test_positive_conversations:
        file_content = file_get_contents(positive_conversation).lower().replace('\n', ' ').replace('\r', '')
        corpus.append(file_content)

    X_test = vectorizer.transform(corpus)

    y_test = oneclass.predict(X_test)
    # print(y_test)
    print(y_test[y_test == 1].size / y_test.size)
