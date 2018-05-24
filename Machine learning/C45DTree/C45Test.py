#-*-coding:utf-8-*-
from numpy import *
from C45DTree import *

dtree=C45DTree()
labels=["age","revenue","student","credit"]
vector1=['0','1','0','0']
vector2=['2','1','1','1']
vector3=['2','1','1','0']
vector4=['1','1','0','0']
vector5=['1','2','1','1']
mytree=dtree.grabTree("data.tree")
print "测试集1：真实输出","no","->","决策树输出",dtree.predict(mytree,labels,vector1)
print "测试集2：真实输出","no","->","决策树输出",dtree.predict(mytree,labels,vector2)
print "测试集3：真实输出","yes","->","决策树输出",dtree.predict(mytree,labels,vector3)
print "测试集4：真实输出","yes","->","决策树输出",dtree.predict(mytree,labels,vector4)
print "测试集5：真实输出","yes","->","决策树输出",dtree.predict(mytree,labels,vector5)

