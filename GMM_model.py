#!/usr/local/bin/python3
#enconding=uft-8
"""
GMM algorithm
"""
import numpy as np
from collections import Counter
import pickle
 
class GMM:
    def __init__(self,K=0,phi=np.array([]),mu=np.array([]),sigma=np.array([])):
        self.K=K               # number of Gaussian components
        self.phi=phi           #1*k  表示每个GMM所占的权重
        self.mu=mu             #n*k  表示k个GMM的均值
        self.sigma=sigma       #k*n*n 表示K个GMM的协方差矩阵，每个都是n*n 

    # k：高斯函数的个数
    # n: 协方差矩阵的大小，即单个样本的维数
    def __init_parameters(self,k,n):   #private attribute
        self.phi=np.random.rand(k)
        self.mu=np.random.rand(n,k)
        self.sigma=np.random.rand(k,n,n)
        for idx in range(k):
            self.sigma[idx]=np.eye(n)       #sigma 初始化为 n*n 的对角阵

    # 根据高斯分布，计算后验概率
    # 每个元素由 高斯分布 生成的概率 (已知高斯分布的类型，每个样本属于该分布的概率)
    # N(x|pMiu,pSigma) = 1/((2pi)^(D/2))*(1/(abs(sigma))^0.5)*exp(-1/2*(x-pMiu)'pSigma^(-1)*(x-pMiu)) 
    def __computeGaussPro(self,X): #compute probability according to Gaussian Distribution
        nDim,nSmp=X.shape          #number of features and samples
        # print('nDim {}, nSmp {}'.format(nDim, nSmp))
        Pro=np.random.rand(self.K,nSmp)
        for idx in range(self.K):
            detSigmaK=np.linalg.det(self.sigma[idx])     #计算协方差矩阵的行列式
            invSigmaK=np.linalg.inv(self.sigma[idx])     #计算协方差矩阵的逆矩阵
            muK=self.mu[:,idx]
            # muK=muK.reshape(nDim,1)
            # print('muK ', muK)
            # print('X ', X)
            for idy in range(nSmp):     #计算每一个样本属于该高斯分布的概率
                sample=X[:,idy]
                sample=sample.reshape(muK.shape)
                diff=sample-muK
                # print('diff ', diff)
                # print('sigma ', invSigmaK)
                # print('value ', np.dot(diff,np.dot(diff,invSigmaK).T))
                Pro[idx,idy]=np.exp(-np.dot(diff,np.dot(diff,invSigmaK).T)/2)
                Pro[idx,idy]/=(np.power(2*np.pi,1/2)*(np.sqrt(detSigmaK)))

        # Modified by 单彦会
        OverFlow = np.ones((self.K,1))*0.00001
        Pro += OverFlow                     #避免分母出现为 0 的情况
        Pro = Pro/np.sum(Pro, axis=0)
        return Pro


    # def __computeGaussPro(self,X): #compute probability according to Gaussian Distribution
    #     nDim,nSmp=X.shape          #number of features and samples
    #     print('nDim {}, nSmp {}'.format(nDim, nSmp))
    #     Pro=np.random.rand(self.K,nSmp)
    #     for idx in range(self.K):
    #         detSigmaK=np.linalg.det(self.sigma[idx])     #计算协方差矩阵的行列式
    #         invSigmaK=np.linalg.inv(self.sigma[idx])     #计算协方差矩阵的逆矩阵
    #         muK=self.mu[:,idx]
    #         for idy in range(nSmp):     #计算每一个样本属于该高斯分布的概率
    #             sample=X[:,idy]
    #             diff=sample-muK
    #             Pro[idx,idy]=np.exp(-np.dot(np.dot(diff,invSigmaK).T,diff)/2)
    #             Pro[idx,idy]/=(np.power(2*np.pi,1/2)*(np.sqrt(detSigmaK)))

    #     # Modified by 单彦会
    #     OverFlow = np.ones((self.K,1))*0.00001
    #     Pro += OverFlow                     #避免分母出现为 0 的情况
    #     Pro = Pro/np.sum(Pro, axis=0)
    #     return Pro

    def __computeLogLikelihood(self,Pro):    #comupte the log likelihood
        val=np.dot(Pro.T,self.phi)
        return np.sum(np.log(val))
 
    def train(self,X):     #N*M ndarray for training GMM,each column is a sample
        N,M=X.shape
        print('input shape ', X.shape)
        if not self.phi.shape[0]*self.mu.shape[0]*self.sigma.shape[0]:
            self.__init_parameters(self.K,N)
        tolerance=1e-3       #threshold of convergence
        convergence=False
        obj_old=-1e10
        obj_his=[]           #stroing the objective function of each iteration
        while not convergence:
            Pro=self.__computeGaussPro(X)
            obj=self.__computeLogLikelihood(Pro)
            diff=obj-obj_old
            print('diff: ', diff)
            if np.fabs(diff)<tolerance:
                break
            obj_old=obj
            obj_his.append(obj)

            #E-step
            Q=(self.phi*Pro.T).T     #used for storing p(z=k|x_i), 为何每行数据都一行，由于Pro问题
            # print("Q shape: ", Q.shape)
            Q/=np.sum(Q,axis=0)

            #M-step
            self.phi=(np.sum(Q.T,axis=0)/M)
            for idx in range(self.K):          #更新均值 和 协方差
                self.mu[:,idx]=np.sum(Q[idx]*X,axis=1)/sum(Q[idx])
                X1=X-self.mu[:,idx].reshape(N,1)     #数据减去均值
                self.sigma[idx]=np.dot(Q[idx]*X1,X1.T)/sum(Q[idx])
        return self.phi,self.mu,self.sigma,obj_his

    def predict(self,testdata):
        pro=self.__computeGaussPro(testdata)
        max_index = np.argmax(pro, axis=0)    
        most_common = Counter(max_index).most_common(1)[0][0]
        return most_common                   #the estimated probability value

    # 存储训练的参数
    def save_parameters(self, filename):
        pars = []
        pars.append(self.K)
        pars.append(self.phi)
        pars.append(self.mu)
        pars.append(self.sigma)
        with open(filename, 'wb') as f:
            pickle.dump(pars, f)
