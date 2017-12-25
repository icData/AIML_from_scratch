#coding=utf-8
# http://www.cnblogs.com/MrLJC/p/4141850.html

from numpy import *

def loadSimpleData():
    dataMat = matrix([[1. , 2.1],
        [2. , 1.1],
        [1.3 , 1.],
        [1. , 1.],
        [2. , 1.]])
    classLabels = [1.0,1.0,-1.0,-1.0,1.0]
    return dataMat, classLabels

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArry = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArry[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArry[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArry

#D是权重向量
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0#在特征所有可能值上遍历
    bestStump = {}#用于存储单层决策树的信息
    bestClasEst = mat(zeros((m,1)))
    minError = inf
    for i in range(n):#遍历所有特征
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshVal = (rangeMin + float(j) * stepSize)#得到阀值
                #根据阀值分类
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr#不同样本的权重是不一样的
                #print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i 
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m =shape(dataArr)[0]
    D = mat(ones((m,1))/m)#初始化所有样本的权值一样
    aggClassEst = mat(zeros((m,1)))#每个数据点的估计值
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        #计算alpha，max(error,1e-16)保证没有错误的时候不出现除零溢出
        #alpha表示的是这个分类器的权重，错误率越低分类器权重越高
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha  
        weakClassArr.append(bestStump)
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) #exponent for D calc, getting messy
        D = multiply(D,exp(expon))                              #Calc New D for next iteration
        D = D/D.sum()
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst
        #print "aggClassEst: ",aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print "total error: ",errorRate
        if errorRate == 0.0: 
            break
    return weakClassArr

#dataToClass 表示要分类的点或点集
def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])#call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst
        print aggClassEst
    return sign(aggClassEst)

def main():
    dataMat,classLabels = loadSimpleData()
    D = mat(ones((5,1))/5)
    classifierArr = adaBoostTrainDS(dataMat,classLabels,30)
    t = adaClassify([0,0],classifierArr)
    print t 
    
if __name__ == '__main__':
    main()
