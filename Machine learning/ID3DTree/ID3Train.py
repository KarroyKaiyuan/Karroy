#-*-coding:utf-8-*-
from numpy import *
from  ID3DTree import *

dtree=ID3DTree()
#[age][revenue][student][credit]
dtree.loadDataSet("dataset.dat",["age","revenue","student","credit"])
dtree.train()


dtree.storeTree(dtree.tree,"data.tree")
mytree=dtree.grabTree("data.tree")
print mytree

