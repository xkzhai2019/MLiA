from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.0],[1.0,1.0],[0,0]])
    labels = ['A','A','B','B']
    return group, labels


def classify0(inputX,dataSet,labels,k): # inputX表示待分类的输入向量，dataSet为训练样本，labels为样本标签，k表示用于投票的邻居节点数目
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inputX,(dataSetSize,1)) - dataSet # 绝对距离
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5                     # 欧式距离
    
    sortedDisIndicies = distances.argsort()         # 从小到大排列的索引 
    
    classCount = {} # 字典
    for i in range(k):
        voteIlabel = labels[sortedDisIndicies[i]]  # 排在第i位的标签
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 # 在对应标签上增加一次计数
        
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)   # 对k个邻居记录的标签数进行从大到小排序
    
    return sortedClassCount[0][0]

def file2matrix(filename): # 从文件中读取数据
    fr = open(filename)
    arrayofLines = fr.readlines()
    numberofLines = len(arrayofLines) # 获得文件行数（样本数）
    returnMat = zeros((numberofLines,3)) # 样本包含3个特征，创建返回的矩阵
    classLabelVector = []
    index = 0
    for line in arrayofLines:
        line = line.strip() # 截取回车字符
        listFromLine = line.split('\t') # 以空格为分隔符做切片
        returnMat[index,:] = listFromLine[0:3] # 将样本数据赋值给矩阵第index行
        classLabelVector.append(int(listFromLine[-1])) # 样本数据最后一位是标签
        
        index += 1
        
    return returnMat, classLabelVector

def autoNorm(dataSet): # 对各特征进行归一化: newValue = (oldValue - min)/(max-min)
    minVals = dataSet.min(0) # 取毎列最小值
    maxVals = dataSet.max(0)
    
    ranges = maxVals - minVals # 数据范围
    normDataSet = zeros(shape(dataSet)) # 存储归一化的数据
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1)) # tile可以将1*3的minVals转换成1000*3的矩阵
    normDataSet = normDataSet / tile(ranges,(m,1)) # /在numpy中表示点除
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.1 # 测试数据占训练样本的百分比
    datingDataMat, datingLabels = file2matrix('./data/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    
    numTestVecs = int(m*hoRatio) # 测试数据的样本数
    errorCount = 0.0
    
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d" %(classifierResult,datingLabels[i]))
        
        if classifierResult != datingLabels[i]:
            errorCount += 1.0 # 如果分类错误，错误+1
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    
def classifyPerson():
    resultList = ['not at all','in small doses', 'in large doses']
    percentTats = float(input("花在游戏上的时间占比？"))
    ffMiles = float(input("每年飞行里程数？"))
    iceCream = float(input("每周吃多少公升冰淇淋？"))
    
    datingDataMat, datingLabels = file2matrix('./data/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    intArray = array([ffMiles,percentTats, iceCream])
    classifierResult = classify0((intArray-minVals)/ranges,normMat,datingLabels,3) # 新的数据也需归一化
    print("这个人的类型是",resultList[classifierResult-1])