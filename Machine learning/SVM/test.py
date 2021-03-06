#################################################
# SVM: support vector machine
# Author : zouxy
# Date   : 2013-12-12
# HomePage : http://blog.csdn.net/zouxy09
# Email  : zouxy09@qq.com
#################################################

from numpy import *
import numpy as np
from SVM import *
import SVM

################## test svm #####################
## step 1: load data
print "step 1: load data..."
dataSet = []
labels = []
#fileIn = open('E:/SVM2/BPData.txt')
fileIn = open('E:/Data_mining/SVM/data.txt')
for line in fileIn.readlines():
    #lineArr = line.strip().split('\t')
    lineArr = line.strip().split(' ')
    dataSet.append([float(lineArr[0]), float(lineArr[1])])
    labels.append(float(lineArr[2]))

dataSet = mat(dataSet)
labels = mat(labels).T
train_x = dataSet[0:81, :]
train_y = labels[0:81, :]
test_x = dataSet[80:101, :]
test_y = labels[80:101, :]

## step 2: training...0.78895625045 1.04006051001	1
print "step 2: training..."
C = 0.6
toler = 0.001
maxIter = 50
svmClassifier = SVM.trainSVM(train_x, train_y, C, toler, maxIter, kernelOption=('linear', 0))
print "b:",svmClassifier.b

## step 3: testing
print "step 3: testing..."
accuracy = SVM.testSVM(svmClassifier, test_x, test_y)

## step 4: show the result
print "step 4: show the result..."
print 'The classify accuracy is: %.3f%%' % (accuracy * 100)
SVM.showSVM(svmClassifier)