
import numpy

#
# Calcular:
#
# accuracy
# precision
# recall
# f1-measure
#

# Accuracy = TP+TN/TP+FP+FN+TN
# Precision = TP/TP+FP
# Recall = TP/TP+FN
# F1 Score = 2*(Recall * Precision) / (Recall + Precision)

# 1 - Ler o arquivo com as predições calculadas: output/pan_prediction-30-70.txt
# 2 - Ler o arquivo "data/pan_br_test.cat"

from sklearn.metrics import confusion_matrix, fbeta_score
from sklearn.metrics import classification_report


def generate_pred(value):

    if value[0] > value[1]:
        return "Positive"
    return "Negative"


classes = ["Positive", "Negative"]

y_pred_file = "./output/pan_prediction.txt"
y_true_file = "./data/pan_br_test.cat"

y_pred = numpy.array(list(map(generate_pred, numpy.loadtxt(open(y_pred_file, "rb"), delimiter=" "))))

with open(y_true_file) as f:

    y_true = numpy.array([x.strip() for x in f.readlines()])

cm = confusion_matrix(y_true, y_pred, labels=classes)

tn, fp, fn, tp = cm.ravel()

print("Confusion Matrix:")
print(cm)
print("Accuracy:")
print((tn+tp)/(tp+fp+fn+tn))
print("Precision:")
precision = tp/(tp+fp)
print(precision)
print("Recall:")
recall = tp/(tp+fn)
print(recall)
print("f1-measure:")
# F1 Score = 2*(Recall * Precision) / (Recall + Precision)
print((2*(recall * precision)) / (recall + precision))
print(fbeta_score(y_true, y_pred, beta=1, labels=classes, average="micro"))
print("f0.5-measure (emphasised Precision with the F with β = 0.5):")
print(fbeta_score(y_true, y_pred, beta=0.5, labels=classes, average="micro"))

print(classification_report(y_true, y_pred, target_names=classes))


# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve









