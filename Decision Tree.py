# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 11:58:27 2018

@author: sepehr
"""
import requests
import pandas as pd
import io
import numpy
import numpy as np  
import matplotlib.pyplot as plt  
names = ['smple','samp2','3','4','5','class']
link = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
csv_text = requests.get(link).text
dataset = pd.read_csv(io.StringIO(csv_text),header=None)
data=np.asarray(dataset)
from sklearn.preprocessing import LabelEncoder
n=LabelEncoder()
label_encoder=n.fit(data[:,0])
print(label_encoder.classes_)
integer_classes=label_encoder.transform(label_encoder.classes_)
print(integer_classes)
sotoon1=label_encoder.transform(data[:,0])
data[:,0]=sotoon1

label_encoder=n.fit(data[:,1])
print(label_encoder.classes_)
integer_classes=label_encoder.transform(label_encoder.classes_)
print(integer_classes)
sotoon2=label_encoder.transform(data[:,1])
data[:,1]=sotoon2
label_encoder=n.fit(data[:,2])
print(label_encoder.classes_)
integer_classes=label_encoder.transform(label_encoder.classes_)
print(integer_classes)
sotoon3=label_encoder.transform(data[:,2])
data[:,2]=sotoon3

label_encoder=n.fit(data[:,3])
print(label_encoder.classes_)
integer_classes=label_encoder.transform(label_encoder.classes_)
print(integer_classes)
sotoon4=label_encoder.transform(data[:,3])
data[:,3]=sotoon4

label_encoder=n.fit(data[:,4])
print(label_encoder.classes_)
integer_classes=label_encoder.transform(label_encoder.classes_)
print(integer_classes)
sotoon5=label_encoder.transform(data[:,4])
data[:,4]=sotoon5

label_encoder=n.fit(data[:,5])
print(label_encoder.classes_)
integer_classes=label_encoder.transform(label_encoder.classes_)
print(integer_classes)
sotoon6=label_encoder.transform(data[:,5])
data[:,5]=sotoon6
X = dataset.iloc[:, :-1].values
#s = X.Series()                
#print(pd.to_numeric(s, errors='coerce'))                
print(X)
y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=0) 
from sklearn.tree import DecisionTreeClassifier    
dtree=DecisionTreeClassifier(max_depth=8,random_state=0)
dtree.fit(X_train,y_train)
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
graph.write_pdf("new.pdf")
graph.write_png("new2.png")
X_global=X_test
y_global=y_test
y_pred = dtree.predict(X_global)

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_global, y_pred))  
print(classification_report(y_global, y_pred))
from sklearn import tree
dtree2 = tree.DecisionTreeClassifier(criterion = "gini", splitter = 'random', max_leaf_nodes = 10, min_samples_leaf = 5, max_depth= 5)
dtree2.fit(X_train,y_train)
DecisionTreeClassifier(criterion = "gini", splitter = 'random', max_depth=5,
             max_leaf_nodes=10, min_samples_leaf=5)
            
predict3 = dtree2.predict(X_train)
print(predict3)
dot_data = StringIO()
export_graphviz(dtree2, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
graph.write_pdf("haras.pdf")
graph.write_png("haras.png")
 

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))
from sklearn import metrics
# Model Accuracy: how often is the classifier correct?

print( "errorrr:",1-metrics.accuracy_score( y_global,y_pred))
from sklearn.metrics import f1_score
print("f1score",f1_score(y_global, y_pred, average='micro'))

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import roc_curve,auc
from scipy import interp
from itertools import cycle
n_classes=3
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(y_global))[:, i], np.array(pd.get_dummies(y_pred))[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
plt.figure(figsize=(8,5))
plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='green', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--',color='red', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.annotate('Random Guess',(.5,.48),color='red')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Naive Bayes - IRIS DATASET')
plt.legend(loc="lower right")
plt.show()

  