# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from sklearn import svm, metrics
import matplotlib.pyplot as plt


# %%
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision


# %%
trainset = torchvision.datasets.MNIST('/home/432/qihaoyu/data/MNIST',train=True,download=False)


# %%
testset = torchvision.datasets.MNIST('/home/432/qihaoyu/data/MNIST',train=False,download=False)


# %%
clf = svm.SVC(cache_size=10000)


# %%
trainset.data.shape


# %%
trainset.targets.shape


# %%
n_samples = len(trainset)
X = trainset.data.reshape(n_samples,-1)
Y = trainset.targets
X.shape,Y.shape


# %%
from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler(feature_range=(0,1)).fit(X)
X_train = scaling.transform(X)
X_test = scaling.transform(testset.data.reshape(len(testset),-1))


# %%
clf.fit(X_train,Y)


# %%
y_test = clf.predict(X_test)


# %%
print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(testset.targets, y_test)))


# %%
print("Confusion matrix:\n%s" % metrics.confusion_matrix(testset.targets, y_test))


# %%
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(clf, X_test, testset.targets)


# %%
y_test_display = clf.predict(X_test[:10,:])
y_test_display,testset.targets[:10]


# %%



