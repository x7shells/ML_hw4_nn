"""
======================================================
Text Classification and ROC
======================================================

Author: Xu He, 2015

"""
print(__doc__)

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from matplotlib.backends.backend_pdf import PdfPages


# load the data set
newsgroup_train = fetch_20newsgroups(subset='train')
newsgroup_test = fetch_20newsgroups(subset='test')

# create TF-IDF feature vectors
count_vect = CountVectorizer(stop_words='english', lowercase=True)
train_counts = count_vect.fit_transform(newsgroup_train.data)

#Term frequency times inverse document frequency
tf_transformer = TfidfTransformer(norm='l2', sublinear_tf=True)
train_tf = tf_transformer.fit_transform(train_counts)

test_counts = count_vect.transform(newsgroup_test.data)
test_tf = tf_transformer.transform(test_counts)

# train the multinomial Naive Bayes
clf_MNB = MultinomialNB(alpha = 0.01)
start1 = time.clock()
clf_MNB.fit(train_tf, newsgroup_train.target)
end1 = time.clock()
time_MNB = end1 - start1

# train the SVM with cosine similarity kernel
clf_SVM = SVC(kernel=cosine_similarity, probability=True)
start2 = time.clock()
clf_SVM.fit(train_tf, newsgroup_train.target)
end2 = time.clock()
time_SVM = end2 - start2

# output predictions on the test data
train_pred_MNB = clf_MNB.predict(train_tf)
pred_MNB = clf_MNB.predict(test_tf)
pred_proba_MNB = clf_MNB.predict_proba(test_tf)

train_pred_SVM = clf_SVM.predict(train_tf)
pred_SVM = clf_SVM.predict(test_tf)
pred_proba_SVM = clf_SVM.predict_proba(test_tf)

# compute the training metrics of the model
train_accuracy_MNB = accuracy_score(newsgroup_train.target, train_pred_MNB)
test_accuracy_MNB = accuracy_score(newsgroup_test.target, pred_MNB)
train_accuracy_SVM = accuracy_score(newsgroup_train.target, train_pred_SVM)
test_accuracy_SVM = accuracy_score(newsgroup_test.target, pred_SVM)

train_precision_MNB = precision_score(newsgroup_train.target, train_pred_MNB)
test_precision_MNB = precision_score(newsgroup_test.target, pred_MNB)
train_precision_SVM = precision_score(newsgroup_train.target, train_pred_SVM)
test_precision_SVM = precision_score(newsgroup_test.target, pred_SVM)

train_recall_MNB = recall_score(newsgroup_train.target, train_pred_MNB)
test_recall_MNB = recall_score(newsgroup_test.target, pred_MNB)
train_recall_SVM = recall_score(newsgroup_train.target, train_pred_SVM)
test_recall_SVM = recall_score(newsgroup_test.target, pred_SVM)

print " -------------------------------------------------------------------------------------------------------"
print "| Machine Learning Methods        |" + "accuracy" + '\t\t|' + "precision" + '\t|' + "recall" + '\t' + '\t|' + "time" + '\t\t|'
print " -------------------------------------------------------------------------------------------------------"
print "| multinomial Naive Bayes | train |" + str(train_accuracy_MNB) + ' \t|' + str(train_precision_MNB) + '\t|' + str(train_recall_MNB) + '\t|' + str(time_MNB) + '\t|'
print " -------------------------------------------------------------------------------------------------------"
print "| multinomial Naive Bayes | test  |" + str(test_accuracy_MNB) + ' \t|' + str(test_precision_MNB) + '\t|' + str(test_recall_MNB) + '\t|' + str(time_MNB) + '\t|'
print " -------------------------------------------------------------------------------------------------------"
print "| SVM & cosine similarity | train |" + str(train_accuracy_SVM) + ' \t|' + str(train_precision_SVM) + '\t|' + str(train_recall_SVM) + '\t|' + str(time_SVM) + '\t|'
print " -------------------------------------------------------------------------------------------------------"
print "| SVM & cosine similarity | test  |" + str(test_accuracy_SVM) + ' \t|' + str(test_precision_SVM) + '\t|' + str(test_recall_SVM) + '\t|' + str(time_SVM) + '\t|'
print " -------------------------------------------------------------------------------------------------------"

binary_target = label_binarize(newsgroup_test.target, classes = np.unique(newsgroup_test.target))

#############################################################################################################
classes = ['comp.graphics', 'comp.sys.mac.hardware', 'rec.motorcycles', 'sci.space', 'talk.politics.mideast']
n_classes = len(classes)

classes_index = []
for c in classes:
	classes_index.append(int(newsgroup_train.target_names.index(c)))

###############################################
# Compute ROC curve and ROC area for each class
fpr_MNB = dict()
tpr_MNB = dict()
roc_auc_MNB = dict()

fpr_SVM = dict()
tpr_SVM = dict()
roc_auc_SVM = dict()

for i in range(n_classes):
    fpr_MNB[i], tpr_MNB[i], _ = roc_curve(binary_target[:, classes_index[i]], pred_proba_MNB[:, classes_index[i]])
    roc_auc_MNB[i] = auc(fpr_MNB[i], tpr_MNB[i])

    fpr_SVM[i], tpr_SVM[i], _ = roc_curve(binary_target[:, classes_index[i]], pred_proba_SVM[:, classes_index[i]])
    roc_auc_SVM[i] = auc(fpr_SVM[i], tpr_SVM[i])

# Plot all ROC curves and save to a PDF file
pp = PdfPages('graphTextClassifierROC.pdf')

plt.figure(1)
for i in range(n_classes):
    plt.plot(fpr_MNB[i], tpr_MNB[i], label='(class = {0}, area = {1})'
                                   ''.format(classes[i], roc_auc_MNB[i]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves of Multinomial Naive Bayes')
plt.legend(loc="best")
pp.savefig()

plt.figure(2)
for i in range(n_classes):
    plt.plot(fpr_SVM[i], tpr_SVM[i], label='(class = {0}, area = {1})'
                                   ''.format(classes[i], roc_auc_SVM[i]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves of SVM with cosine kernel')
plt.legend(loc="best")
pp.savefig()

plt.show()
pp.close()





        