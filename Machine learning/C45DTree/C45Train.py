#-*-coding:utf-8-*-
from numpy import *
from C45DTree import *

dtree=C45DTree()
#[age][revenue][student][credit]
dtree.loadDataSet("dataset.dat",["age","revenue","student","credit"])
dtree.train()


dtree.storeTree(dtree.tree,"data.tree")
mytree=dtree.grabTree("data.tree")
print mytree

