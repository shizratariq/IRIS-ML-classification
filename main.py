from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
iris= datasets.load_iris()
features= iris.data
labels=iris.target
#print(features[0],labels[0])

clf=KNeighborsClassifier()
clf.fit(features,labels)
preds= clf.predict([[5.1,3.5,1.4,0.2]])
print(preds)