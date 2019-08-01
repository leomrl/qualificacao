#
# https://cs230-stanford.github.io/train-dev-test-split.html
#
import random

import numpy
import shutil

from util import cleanup

import xml.etree.ElementTree as ET

from sklearn.model_selection import StratifiedKFold

K_VALUE = 5

result_dir = "./result-br"
kfold_dir = "/k_fold"

result_kfold_dir = result_dir + kfold_dir

cleanup(result_kfold_dir)
shutil.rmtree(result_kfold_dir)

for index in range(1, K_VALUE + 1):
    cleanup(result_kfold_dir + '/' + str(index))

corpus_predatory_file = "/home/leonardo/Documents/Workspace/Quali/data/original-pan12/pan12-br-sexual-predator-only-training-corpus.xml"
corpus_non_predatory_file = "/home/leonardo/Documents/Workspace/Quali/data/original-pan12/pan12-br-sexual-predator-identification-training-corpus.xml"
corpus_sexual_predator_ids_file = "/home/leonardo/Documents/Workspace/Quali/data/original-pan12/pan12-br-sexual-predator-identification-training-corpus-predators.txt"

corpus_train_file = 'pan12-br-sexual-predator-identification-training-corpus.xml'
corpus_test_file = 'pan12-br-sexual-predator-identification-test-corpus.xml'


xml_predatory_file = open(corpus_predatory_file, "r", encoding="utf-8")
predatory_corpus = ET.parse(xml_predatory_file)

xml_non_predatory_file = open(corpus_non_predatory_file, "r", encoding="utf-8")
non_predatory_corpus = ET.parse(xml_non_predatory_file)

predatory_conversations_ids = predatory_corpus.findall("conversation[@id]")
predatory_conversations_ids = numpy.array([1000 * int(d.attrib['id'], 10) for d in predatory_conversations_ids])
predatory_conversations_classes = numpy.array([0] * len(predatory_conversations_ids))

non_predatory_conversations_ids = non_predatory_corpus.findall("conversation[@id]")
non_predatory_conversations_ids = numpy.array([int(d.attrib['id'], 10) for d in non_predatory_conversations_ids])
non_predatory_conversations_classes = numpy.array([1] * len(non_predatory_conversations_ids))

X = numpy.concatenate([predatory_conversations_ids, non_predatory_conversations_ids])
y = numpy.concatenate([predatory_conversations_classes, non_predatory_conversations_classes])

print(X)
print(y)

skf = StratifiedKFold(n_splits=5, shuffle=True)
skf.get_n_splits(X, y)

print(skf)

index = 1


def generate_dataset(index, X_train, X_test):
    root_train_set = ET.Element("conversations")
    root_test_set = ET.Element("conversations")

    result_train_corpus = result_kfold_dir + '/' + str(index) + '/' + corpus_train_file
    result_test_corpus = result_kfold_dir + '/' + str(index) + '/' + corpus_test_file

    for conversation_index in range(0, len(X_train)):

        conversation_train_id = X_train[conversation_index]

        if conversation_train_id >= 1000:
            conversation_id = conversation_train_id // 1000
            conversation = predatory_corpus.find(".//conversation[@id='" + str(conversation_id) + "']")
            root_train_set.append(conversation)

        else:
            conversation_id = X_train[conversation_index]
            conversation = non_predatory_corpus.find(".//conversation[@id='" + str(conversation_id) + "']")
            root_train_set.append(conversation)

    ET.ElementTree(root_train_set).write(result_train_corpus, encoding="utf-8", xml_declaration=True)

    for conversation_index in range(0, len(X_test)):

        conversation_test_id = X_test[conversation_index]

        if conversation_test_id >= 1000:
            conversation_id = conversation_test_id // 1000
            conversation = predatory_corpus.find(".//conversation[@id='" + str(conversation_id) + "']")
            root_test_set.append(conversation)

        else:
            conversation_id = X_test[conversation_index]
            conversation = non_predatory_corpus.find(".//conversation[@id='" + str(conversation_id) + "']")
            root_test_set.append(conversation)

    ET.ElementTree(root_test_set).write(result_test_corpus, encoding="utf-8", xml_declaration=True)

    return


for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)

    X_train, X_test = X[train_index], X[test_index]
    print("X_train:", X_train, "X_test:", X_test)
    generate_dataset(index, X_train, X_test)
    index += 1

    # y_train, y_test = y[train_index], y[test_index]
    # print("y_train:", y_train, "y_test:", y_test)
