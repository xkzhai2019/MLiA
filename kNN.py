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

def file2matrix(filename):
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