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