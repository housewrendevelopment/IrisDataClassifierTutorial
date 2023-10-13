from scipy.spatial import distance

def euc(a,b):
  return distance.euclidean(a,b)

import random

class ScrappyKNN():
    def fit (self, X_train, y_train):
      self.X_train = X_train
      self.y_train = y_train
      
    def predict(self, X_test):
      predictions = []
      for row in X_test:
        #label = random.choice(self.y_train)
        label = self.closest(row)
        predictions.append(label)
      return predictions

    def closest(self, row):
      best_dist = euc(row, self.X_train[0])
      best_index = 0
      for i in range (1, len(self.X_train)):
        dist = euc(row, self.X_train[i])
        if dist < best_dist:
          best_dist = dist
          best_index = i
      return self.y_train[best_index]

import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()

#print(iris.feature_names)
#print(iris.target_names)
#print(iris.data[0])
#print(iris.target[0])
#for i in range (len(iris.target)):
#  print("Example %d: label %s, feature %s"%(i, iris.target[i], iris.data[i]))

test_idx = [0,50,100]

#training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

#print(test_target)
#print(clf.predict(test_data))

#viz code for easy to read PDF of decision tree
#from six import StringIO
#import pydot
#dot_data = StringIO()
#tree.export_graphviz(clf, out_file=dot_data,
#                    feature_names=iris.feature_names,
#                    class_names=iris.target_names,
#                    filled=True, rounded=True, impurity=False)
#graph = pydot.graph_from_dot_data(dot_data.getvalue())
#graph[0].write_pdf("iris.pdf")

X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

#my_classifier = tree.DecisionTreeClassifier()
#from sklearn.neighbors import KNeighborsClassifier
#my_classifier = KNeighborsClassifier()

#writing our own classifier
my_classifier = ScrappyKNN()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)
#print(predictions)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))