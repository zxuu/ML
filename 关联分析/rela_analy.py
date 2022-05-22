import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def loadDataSet():
    '''
    构建数据集
    :return: 数据集
    '''
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def creatC1(dataSet):
    '''
    构建1项数据集
    :param dataSet: 原始数据集
    :return: 1项数据集
    '''
    c1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in c1:
                c1.append([item])
    c1.sort()
    return list(map(frozenset,c1))

def scanD(D,CK,minSupport):
    '''
    扫描dataSet
    :param D: dataSet
    :param CK: 项集
    :param minSupport: 最小支持度
    :return: 满足条件的项，所有的项和支持度
    '''
    ssCnt = {}
    for tid in D: #遍历dataset
        for can in CK: #遍历项集
            if can.issubset(tid):
                if not can in ssCnt: #如果项集不在字典当中则创建一个键
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1 #否则值加一
    numItems = float(len(list(D))) #数据总长度
    retlist = []
    supportData = {}
    for key in ssCnt: #遍历键值
        support = ssCnt[key]/numItems
        if support >= minSupport: #如果大于支持度阈值
            retlist.insert(0,key) #满足条件的项留下来（仅记录项）
        supportData[key] = support #所有的项和支持度{项：支持度}
    return retlist, supportData

def aprioriGen(LK,k):
    '''
    拼接项
    :param LK: 项集
    :param K: k值
    :return: 拼接的项集
    '''
    retlist = []
    lenLK = len(LK)
    for i in range(lenLK):
        for j in range(i+1,lenLK):
            L1 = list(LK[i])[:k-2]
            L2 = list(LK[j])[:k-2]
            if L1 == L2:
                retlist.append(LK[i] | LK[j])
    return retlist

def apriori(dataSet,minSupport=0.5):
    '''
    apriori算法
    :param dataSet: 原始数据集
    :param minSupport: 最小支持度
    :return: 项集，所有项集的支持度
    '''
    C1 = creatC1(dataSet) #创建一项集
    L1,supportData = scanD(dataSet,C1,minSupport) #扫描一项集
    L = [L1] #满足条件的一项集
    k = 2
    while(len(L[k-2]) > 0): #如果项集还可以拼接
        Ck = aprioriGen(L[k-2],k) #拼接
        LK,supk = scanD(dataSet,Ck,minSupport) #扫描项集
        supportData.update(supk)
        L.append(LK) #添加扫描后的项集.第一行一项集、第二行二项集......
        k += 1
    return L,supportData


def calConf(freqSet, H, supportData, rulelist, minConf):
    prunedh = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minConf:
            print(freqSet-conseq,'-->',conseq,'conf:',conf)
            rulelist.append((freqSet-conseq,conseq,conf))
            prunedh.append(conseq)
    return prunedh


def rulessFromConseq(freqSet, H, supportData, rulelist, minConf=0.6):
    '''

    :param L: 所有满足条件的项集,2项集、3项集、、、
    :param H1: 项集里的单个元素
    :param supportData: 所有项集和支持度
    :param rulelist:
    :param minConf: 最小置信度
    :return:
    '''
    m = len(H[0])
    while(len(freqSet) > m):
        H = calConf(freqSet,H,supportData,rulelist,minConf)
        if (len(H)>1):
            aprioriGen(H,m+1)
            m += 1
        else:
            break

def generateRules(L,supportData,minConf = 0.6):
    '''

    :param L: 所有满足条件的项集
    :param supportData: 所有项集和支持度
    :param minConf: 最小置信度
    :return:
    '''
    rulelist = []
    for i in range(1,len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            rulessFromConseq(freqSet,H1,supportData,rulelist,minConf)


if __name__ == '__main__':
    dataSet = loadDataSet()
    L,support = apriori(dataSet)
    i = 0
    for freq in L:
        print('项数',i+1,':',freq)
        i += 1
    rules = generateRules(L,support,minConf=0.5)