import time

import os

import glob

import xml.etree.cElementTree as ET

from util import cleanup
from util import file_get_contents

import random

start_time = time.time()

result_dir = './result'

base_dir = '/home/leonardo/Documents/Workspace/Quali'

base_dir_br = './result-br'  # + '/k_fold' + '/5'

#Flag para escolha do dataset:

raw_dataset = "pan-12-br"  # [ "clef" || "pan-12-br" ]

if raw_dataset is "clef":

    dataset_train_dir = base_dir + '/data/clef/training'
    dataset_test_dir = base_dir + '/data/clef/test'

    corpus_training_file = dataset_train_dir + '/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml'
    corpus_training_predator_id_file = dataset_train_dir + '/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt'

    corpus_test_file = dataset_test_dir + '/pan12-sexual-predator-identification-test-corpus-2012-05-17.xml'
    corpus_test_predator_id_file = dataset_test_dir + '/pan12-sexual-predator-identification-groundtruth-problem1.txt'

    result_dir = result_dir + '/data'
else:

    dataset_train_dir = base_dir + '/data/original-pan-12-split'
    dataset_test_dir = base_dir + '/data/original-pan-12-split'

    corpus_training_file = base_dir_br + '/pan12-br-sexual-predator-identification-training-corpus.xml'
    corpus_training_predator_id_file = dataset_train_dir + '/pan12-br-sexual-predator-identification-training-corpus-predators.txt'

    corpus_test_file = base_dir_br + '/pan12-br-sexual-predator-identification-test-corpus.xml'
    corpus_test_predator_id_file = dataset_test_dir + '/pan12-br-sexual-predator-identification-training-corpus-predators.txt'

    result_dir = './data'

#Variaveis Parte 1

dataset_result_train_dir = result_dir + '/train'
dataset_result_test_dir = result_dir + '/test'

#Variaveis Parte 2

dataset_train_result_file = result_dir + '/pan_br_train.txt.tok'
dataset_test_result_file = result_dir + '/pan_br_test.txt.tok'

dataset_train_cat_result_file = result_dir + '/pan_br_train.cat'
dataset_test_cat_result_file = result_dir + '/pan_br_test.cat'

dataset_class_dic_file = result_dir + '/pan_br.dic'

dataset_train = []
dataset_test = []


dataset_train_cat = []
dataset_test_cat = []

category_positive = "Positive"
category_negative = "Negative"


dataset_dirs = [dataset_train_dir, dataset_test_dir]
category_dirs = [category_positive, category_negative]

#Parte 1: Extrair dados do PAN-2012 e gerar os dados no mesmo formato de ebrahimi.
#A Parte 2 ficará padronizada além de servir para gerar os dados com o dataset BR.

cleanup(result_dir)
cleanup(dataset_result_train_dir)
cleanup(dataset_result_train_dir + '/' + category_positive)
cleanup(dataset_result_train_dir + '/' + category_negative)
cleanup(dataset_result_test_dir)
cleanup(dataset_result_test_dir + '/' + category_positive)
cleanup(dataset_result_test_dir + '/' + category_negative)

corpus_files = [corpus_training_file, corpus_test_file]
predator_id_files = [corpus_training_predator_id_file, corpus_test_predator_id_file]


def is_predatory_conversation(conversation):

    is_predatory_conversation = False

    messages_per_conversation = conversation.findall("message")
    for message in messages_per_conversation:
        author = message.find("author")

        if author.text in sexual_predators_ids:

            is_predatory_conversation = True
            break

    #print("is_predatory_conversation = ", is_predatory_conversation)

    return is_predatory_conversation


def get_conversation_content(conversation):

    messages_per_conversation = conversation.findall("message")

    conversation_content = ''

    for message in messages_per_conversation:
        text = message.find("text")

        if not (text.text is None):
            conversation_content += text.text + " "

    return conversation_content


def save_message(index, conversation_id, is_predatory, conversation_messages_content):

    conversation_result_dir = dataset_result_train_dir if index == 0 else dataset_result_test_dir

    if is_predatory:
        conversation_destination = conversation_result_dir + '/' + category_positive + '/' + conversation_id + '.txt'
    else:
        conversation_destination = conversation_result_dir + '/' + category_negative + '/' + conversation_id + '.txt'

    #print(conversation_destination)

    with open(conversation_destination, "w", encoding="utf-8") as f:
        f.seek(0)
        f.write("%s\n" % conversation_messages_content)
    f.close()


for index in 0,  1:
    current_corpus_file = corpus_files[index]
    current_predator_id_file = predator_id_files[index]

    print(current_corpus_file)
    print(current_predator_id_file)

    sexual_predators_ids_file = open(current_predator_id_file, "r")
    sexual_predators_ids = sexual_predators_ids_file.read().splitlines()
    sexual_predators_ids_file.close()

    xml_file = open(current_corpus_file, "r", encoding="utf8")

    current_parsed_corpus_file = ET.parse(xml_file)
    conversations = current_parsed_corpus_file.findall("conversation")

    for conversation in conversations:

        conversation_id = conversation.get("id")

        is_predatory = is_predatory_conversation(conversation)
        conversation_messages_content = get_conversation_content(conversation)

        save_message(index, conversation_id, is_predatory, conversation_messages_content)

    xml_file.close()



#Parte 2: Gerar input para conText 2.0

# gen_vocab parameters "LowerCase UTF8"

# conText 4.0 gen_regions guidelines (extracted from manual).

#
# Tokenized text file (input) The input text file format is the same as gen vocab above. The file should contain one
# document per line, and each document should be already tokenized so that tokens are delimited by space.
#

# OK
# Label file (input) Each line of a label file should contain classification labels for each document, and the order of
# the documents must be the same as the tokenized text file. In case of multi-label classification (i.e., more than one
# label can be assigned to each document), the labels should be delimited by a vertical line |.
# See examples/data/s-multilab-test.cat for example. The labels should be strings without white space,
# and they must be declared in a label dictionary described below.
#

# OK
# Text/label file naming conventions The tokenized text file and the corresponding label file must have the same path-
# name stem with different file extensions, .txt.tok and .cat, respectively, by default, e.g., text file data/s-train.txt.tok
# and label file data/s-train.cat.
#

# OK
# Label dictionary file (input) The labels used in the label file above must be declared in a label dictionary file. The
# label dictionary file should contain one label per line. See examples/data/s-cat.dic for example.
#

# OK
# Vocabulary file (input) The vocabulary file should be generated by gen vocab above. To use a vocabulary file
# generated by some other means, note that a tab character (0x09) is regarded as a delimiter, i.e., in each line, a tab char-
# acter and anything that follows are ignored. Also note that the case option must be consistent; e.g., if gen region
# specifies LowerCase (which converts upper-case letters to lower-case), then the contents of the vocabulary file must
# also be all lower-case.
#



dataset_dirs = [dataset_result_train_dir, dataset_result_test_dir]

_label_train_fn = open(dataset_train_cat_result_file, "w+", encoding="utf-8")
_label_test_fn = open(dataset_test_cat_result_file, "w+", encoding="utf-8")
_input_train_fn = open(dataset_train_result_file, "w+", encoding="utf-8")
_input_test_fn = open(dataset_test_result_file, "w+", encoding="utf-8")


for dataset_dir in dataset_dirs:
    print(dataset_dir)

    _input_fn = _input_train_fn if "test" not in dataset_dir else _input_test_fn
    _label_fn = _label_train_fn if "test" not in dataset_dir else _label_test_fn

    conversations_category_hash = {}

    dataset_dir_hash = {category_positive: {}, category_negative: {}}

    for category_dir in category_dirs:

        print(category_dir)
        unsorted_conversations_file = glob.glob(os.path.join((dataset_dir + '/' + category_dir + '/'), '*.txt'))
        sorted_conversations_file = sorted(unsorted_conversations_file)
        print(len(sorted_conversations_file))
        for conversation_file in sorted_conversations_file:
            #_label_fn.write(category_dir+"\r\n")
            #print(conversation_file)
            file_content = file_get_contents(conversation_file).lower().replace('\n', ' ').replace('\r', '')
            dataset_dir_hash[category_dir][conversation_file] = {conversation_file: file_content}
            #_input_fn.write(file_content+"\r\n")

        conversations_category_hash = {**conversations_category_hash, **dataset_dir_hash[category_dir]}

    dataset_dir_keys = list(conversations_category_hash.keys())

    random.shuffle(dataset_dir_keys)

    for key in dataset_dir_keys:

        label = os.path.basename(os.path.dirname(key))

        _label_fn.write(label+"\r\n")

        file_content = file_get_contents(key).lower().replace('\n', ' ').replace('\r', '')
        _input_fn.write(file_content + "\r\n")

    _label_fn.close()

_input_train_fn.close()
_input_test_fn.close()


_label_dic_fn = open(dataset_class_dic_file, "w+")
for i in category_dirs:
    _label_dic_fn.write(i+"\r\n")
_label_dic_fn.close()


