from math import log

def calcShannonEnt(dataSet):
    numEntries = len(dataSet) # 总数据量
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): # 创建数据字典，键值为最后一列数值
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1 # 键值记录当前类别出现的次数
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:  # 找到值与想要划分的特征值相同样本，并去除该特征，得到划分后的数据
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1 # 计算特征数量
    baseEntropy = calcShannonEnt(dataSet) # 计算初始熵
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        featList  = [example[i] for example in dataSet] # 取出数据集中第i个特征所有值
        uniqueVals = set(featList) # 去除重复值
        newEntropy = 0.0
        for value in uniqueVals:   # 依次按照不同的值进行抽取
            subDataSet = splitDataSet(dataSet,i,value)  
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


