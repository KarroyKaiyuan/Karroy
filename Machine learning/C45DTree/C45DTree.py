#-*-coding:utf-8-*-
from numpy import *
import math
import copy
import cPickle as pickle

class C45DTree(object):

    def __init__(self):
        self.tree={}
        self.dataSet=[]
        self.labels=[]
    #数据导入函数
    def loadDataSet(self,path,labels):
        recordlist=[]
        fp=open(path,"rb")
        content=fp.read()
        fp.close()
        rowlist=content.splitlines()
        recordlist=[row.split("\t") for row in rowlist if row.strip()]
        self.dataSet=recordlist
        self.labels=labels
    #执行决策树函数
    def train(self):
            labels=copy.deepcopy(self.labels)
            self.tree=self.buildTree(self.dataSet,labels)

    #构建决策树1111111111111111111111111111111111111111
    def buildTree(self,dataSet,labels):
        cateList=[data[-1] for data in dataSet]
        if cateList.count(cateList[0])==len(cateList):
            return cateList[0]
        if len(dataSet[0])==1:
            return self.maxCate(cateList)
        #算法核心

        bestFeat,featValueList=self.getBestFeat(dataSet)
        bestFeatLabel=labels[bestFeat]
        tree={bestFeatLabel:{}}
        del(labels[bestFeat])
        for value in featValueList:
            sublabels=labels[:]
            #按最优特征列和值分割数据集
            splitDataset=self.splitDataSet(dataSet,bestFeat,value)
            subTree=self.buildTree(splitDataset,sublabels)
            tree[bestFeatLabel][value]=subTree
        return tree
    #计算出现次数最多的类别标签
    def maxCate(self,catelist):
        items=dict([(catelist.count(i),i) for i in catelist])
        return items[max(items.keys())]
    #计算最优特征11111111111111111111111111
    def getBestFeat(self,dataSet):
        Num_Feats=len(dataSet[0][:-1])
        totality=len(dataSet)
        BaseEntropy=self.computeEntropy(dataSet)
        ConditionEntroy=[]
        slpitInfo=[]
        allFeatVList=[]
        for f in xrange(Num_Feats):
            featList=[example[f] for example in dataSet]
            [splitI,featureValueList]=self.computeSplitInfo(featList)
            allFeatVList.append(featureValueList)
            slpitInfo.append(splitI)
            resultGain=0.0
            for value in featureValueList:
                subSet=self.splitDataSet(dataSet,f,value)
                appearNum=float(len(subSet))
                subEntropy=self.computeEntropy(subSet)
                resultGain+=(appearNum/totality)*subEntropy
            ConditionEntroy.append(resultGain)
        infoGainArray=BaseEntropy*ones(Num_Feats)-array(ConditionEntroy)
        infoGainRatio=infoGainArray/array(slpitInfo)
        bestFeatureIndex=argsort(-infoGainRatio)[0]
        return bestFeatureIndex,allFeatVList[bestFeatureIndex]

    #计算划分信息
    def computeSplitInfo(self,featureVList):
        numEntries=len(featureVList)
        featureVauleSetList=list(set(featureVList))
        valueCounts=[featureVList.count(featVec) for featVec in featureVauleSetList]
        pList=[float(item)/numEntries for item in valueCounts]
        lList=[item*math.log(item,2) for item in pList]
        splitInfo=-sum(lList)
        return  splitInfo,featureVauleSetList
    #计算最优特征
    '''
    def getBestFeat(self,dataSet):
        numFeatures=len(dataSet[0])-1
        baseEntropy=self.computeEntropy(dataSet)
        bestInfoGain=0.0
        bestFeature=-1
        for i in xrange(numFeatures):
            uniqueVals=set([data[i] for data in dataSet])
            newEntropy=0.0
            for value in uniqueVals:
                subDataSet=self.splitDataSet(dataSet,i,value)
                prob=len(subDataSet)/float(len(dataSet))
                newEntropy+=prob*self.computeEntropy(subDataSet)
            infoGain=baseEntropy-newEntropy
            if(infoGain>bestInfoGain):
                bestInfoGain=infoGain
                bestFeature=i
        return bestFeature
'''

    # 计算信息熵
    def computeEntropy(self, dataSet):
        datalen = float(len(dataSet))
        cateList = [data[-1] for data in dataSet]
        items = dict([(i, cateList.count(i)) for i in cateList])
        infoEntropy = 0.0
        for key in items:
            prob = float(items[key]) / datalen
            infoEntropy -= prob * math.log(prob, 2)
        return infoEntropy
    #划分数据集
    def splitDataSet(self,dataSet,axis,value):
        rtnList=[]
        for featVec in dataSet:
            if featVec[axis]==value:
                rFeatVec=featVec[:axis]
                rFeatVec.extend(featVec[axis+1:])
                rtnList.append(rFeatVec)
        return rtnList



    #持久化决策树
    def storeTree(self,inputTree,filename):
        fw=open(filename,'w')
        pickle.dump(inputTree,fw)
        fw.close()
    def grabTree(self,filename):
        fr=open(filename)
        return pickle.load(fr)
    #决策树分类
    def predict(self,inputTree,featLabels,testVec):
        root=inputTree.keys()[0]
        secondDict=inputTree[root]
        featIndex=featLabels.index(root)
        key=testVec[featIndex]
        valueOfFeat=secondDict[key]
        if isinstance(valueOfFeat,dict):
            classLabel=self.predict(valueOfFeat,featLabels,testVec)
        else:classLabel=valueOfFeat
        return classLabel