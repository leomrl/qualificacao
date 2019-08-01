#
# https://cs230-stanford.github.io/train-dev-test-split.html
#
import random

from util import cleanup

import xml.etree.ElementTree as ET

result_dir = "./result-br"

cleanup(result_dir)

corpus_predatory_file = "/home/leonardo/Documents/Workspace/Quali/data/original-pan12/pan12-br-sexual-predator-only-training-corpus.xml"
corpus_non_predatory_file = "/home/leonardo/Documents/Workspace/Quali/data/original-pan12/pan12-br-sexual-predator-identification-training-corpus.xml"
corpus_sexual_predator_ids_file = "/home/leonardo/Documents/Workspace/Quali/data/original-pan12/pan12-br-sexual-predator-identification-training-corpus-predators.txt"


corpus_train_file = 'pan12-br-sexual-predator-identification-training-corpus.xml'
corpus_test_file = 'pan12-br-sexual-predator-identification-test-corpus.xml'
result_train_corpus = result_dir + '/' + corpus_train_file
result_test_corpus = result_dir + '/' + corpus_test_file

xml_predatory_file = open(corpus_predatory_file, "r", encoding="utf-8")
predatory_corpus = ET.parse(xml_predatory_file)

xml_non_predatory_file = open(corpus_non_predatory_file, "r", encoding="utf-8")
non_predatory_corpus = ET.parse(xml_non_predatory_file)


predatory_conversations = predatory_corpus.findall("conversation")
non_predatory_conversations = non_predatory_corpus.findall("conversation")

random.seed(230)
random.shuffle(predatory_conversations)
random.shuffle(non_predatory_conversations)

split_ratio = 0.3


#Train set
split_predatory = int(split_ratio * len(predatory_conversations))
train_predatory = predatory_conversations[:split_predatory]
test_predatory = predatory_conversations[split_predatory:]

#Test set
split_non_predatory = int(split_ratio * len(non_predatory_conversations))
train_non_predatory = non_predatory_conversations[:split_non_predatory]
test_non_predatory = non_predatory_conversations[split_non_predatory:]



#Train Set Generation

merged_train_set = train_predatory + train_non_predatory

random.shuffle(merged_train_set)


root_train_set = ET.Element("conversations")

for conversation in merged_train_set:
    root_train_set.append(conversation)

ET.ElementTree(root_train_set).write(result_train_corpus, encoding="utf-8", xml_declaration=True)



#Test Set Generation

merged_test_set = test_predatory + test_non_predatory

random.shuffle(merged_test_set)




root_test_set = ET.Element("conversations")

for conversation in merged_test_set:
    root_test_set.append(conversation)

ET.ElementTree(root_test_set).write(result_test_corpus, encoding="utf-8", xml_declaration=True)


