from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from math import log
import operator
import pickle

def createDataSet():
	dataSet = [[0, 0, 0, 0, 'no'],
			[0, 0, 0, 1, 'no'],
			[0, 1, 0, 1, 'yes'],
			[0, 1, 1, 0, 'yes'],
			[0, 0, 0, 0, 'no'],
			[1, 0, 0, 0, 'no'],
			[1, 0, 0, 1, 'no'],
			[1, 1, 1, 1, 'yes'],
			[1, 0, 1, 2, 'yes'],
			[1, 0, 1, 2, 'yes'],
			[2, 0, 1, 2, 'yes'],
			[2, 0, 1, 1, 'yes'],
			[2, 1, 0, 1, 'yes'],
			[2, 1, 0, 2, 'yes'],
			[2, 0, 0, 0, 'no']]
	labels = ['F1-AGE', 'F2-WORK', 'F3-HOME', 'F4-LOAN']
	return dataSet, labels


def majorityCnt(classList):
    '''
    返回最多的类别
    :param classList: 数据data的标签值
    :return: 最多的标签
    '''
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote]=0
        classCount[vote] += 1
    sortedclassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedclassCount[0][0]


def calcshannonEnt(dataset):
    '''
    计算熵值
    :param dataset: 数据
    :return: 熵值
    '''
    num_examples = len(dataset)    #数据量
    labelCounts = {}    #各个标签的数量
    for featVec in dataset:
        currentlabel = featVec[-1]
        if currentlabel not in labelCounts.keys():
            labelCounts[currentlabel] = 0
        labelCounts[currentlabel] += 1

    #计算熵值
    shannonEnt = 0
    for key in labelCounts:
        prop = float(labelCounts[key])/num_examples
        shannonEnt -= prop*log(prop,2)
    return shannonEnt


def splitDataSet2(dataset, i, val):
    '''
    几个特征逐一分析
    :param dataset: 原始数据
    :param i: 第i列数据
    :param val: 第i列数据的一个特征
    :return: 切分后的数据
    '''
    retDataSet = []
    for featVec in dataset:
        if featVec[i] == val:
            reducedFeatVec = featVec[:i]
            reducedFeatVec.extend(featVec[i+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataset):
    '''
    选择最好的特征
    :param dataset: 数据
    :return: 选择最好的特征(下标)
    '''
    num_features = len(dataset[0]) -1    #特征的数量
    baseEntropy = calcshannonEnt(dataset)    #计算初始熵值

    bestInfoGain = 0  #最好的信息增益
    bestFeature = -1    #最好的特征

    #计算各个熵值
    for i in range(num_features):
        featList = [example[i] for example in dataset]    #选取第i列的数据
        uniqueValues = set(featList)    #第i列的数据中的不重复的数据
        newEntropy = 0
        for val in uniqueValues:    #几个特征逐一分析
            subDataSet = splitDataSet2(dataset,i,val)
            prob = len(subDataSet)/float(len(dataset))    #概率值
            newEntropy += prob*calcshannonEnt(subDataSet)    #计算新的熵值
        infoGain = baseEntropy - newEntropy    #计算信息增益
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def splitDataSet(dataset, bestFeat, value):
    '''
    分割数据
    :param dataset: 数据
    :param bestFeat: 删掉这一列
    :param value:
    :return: 分割后的数据
    '''


def createTree(dataset,labels,featLabels):
    '''
    递归选择节点
    :param dataset: 数据
    :param labels: 标签
    :param featLabels: 记录选择的节点
    :return: null
    '''
    classList = [example[-1] for example in dataset]
    #如果是叶节点（数据熵值为0）
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #判断是否特征已经删完了
    if len(classList[0]) == 1:
        return majorityCnt(classList)     #返回最多的类别

    bestFeat = chooseBestFeatureToSplit(dataset)    #选择特征去分割,下标

    bestFeatLabel = labels[bestFeat]
    featLabels.append(bestFeatLabel)    #按顺序添加特征

    myTree = {bestFeatLabel:{}}
    del labels[bestFeat]    #删除节点标签

    featValue = [example[bestFeat] for example in dataset]
    uniqueValues = set(featValue)    #当前列有几个不同的值

    for value in uniqueValues:
        sublabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet2(dataset,bestFeat,value),sublabels,featLabels)
    return myTree


'''----------------------------画图模块-------------------------------------------------------'''
def getNumLeafs(myTree):
	numLeafs = 0
	firstStr = next(iter(myTree))
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
	    if type(secondDict[key]).__name__=='dict':
	        numLeafs += getNumLeafs(secondDict[key])
	    else:   numLeafs +=1
	return numLeafs


def getTreeDepth(myTree):
	maxDepth = 0
	firstStr = next(iter(myTree))
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
	    if type(secondDict[key]).__name__=='dict':
	        thisDepth = 1 + getTreeDepth(secondDict[key])
	    else:   thisDepth = 1
	    if thisDepth > maxDepth: maxDepth = thisDepth
	return maxDepth

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
	arrow_args = dict(arrowstyle="<-")
	font = FontProperties(fname=r"c:\windows\fonts\simsunb.ttf", size=14)
	createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
		xytext=centerPt, textcoords='axes fraction',
		va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, FontProperties=font)


def plotMidText(cntrPt, parentPt, txtString):
	xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
	yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
	createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
	decisionNode = dict(boxstyle="sawtooth", fc="0.8")
	leafNode = dict(boxstyle="round4", fc="0.8")
	numLeafs = getNumLeafs(myTree)
	depth = getTreeDepth(myTree)
	firstStr = next(iter(myTree))
	cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
	plotMidText(cntrPt, parentPt, nodeTxt)
	plotNode(firstStr, cntrPt, parentPt, decisionNode)
	secondDict = myTree[firstStr]
	plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':
			plotTree(secondDict[key],cntrPt,str(key))
		else:
			plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
			plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
			plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
	plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
def createPlot(inTree):
	fig = plt.figure(1, facecolor='white')													#创建fig
	fig.clf()																				#清空fig
	axprops = dict(xticks=[], yticks=[])
	createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    							#去掉x、y轴
	plotTree.totalW = float(getNumLeafs(inTree))											#获取决策树叶结点数目
	plotTree.totalD = float(getTreeDepth(inTree))											#获取决策树层数
	plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;								#x偏移
	plotTree(inTree, (0.5,1.0), '')															#绘制决策树
	plt.show()

if __name__ == '__main__':
    dataSet, labels = createDataSet()
    featLabels = []
    myTree = createTree(dataSet,labels,featLabels)

    createPlot(myTree)
