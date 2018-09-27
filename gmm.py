"""
部分参考：https://github.com/stober/gmm
"""

import time
import random
import numpy as np
import numpy.linalg as la

try:
    from kmeans import Kmeans
except:
    import scipy.cluster.vq as vq



class Guass_distribution:
    """
    属性值：
    d     高斯分布的维度
    mu    均值 ndarray 形状为 (d,1)
    Sigma 协方差矩阵 ndarray 形状为 (d,d)
    weight此高斯分布的权重
    """
    def __init__(self,mu=[[0,0]],Sigma=[[1,0],[0,1]],dim=2):
        self.mu = np.array(mu)
        self.Sigma = np.array(Sigma)
        self.dim = dim
        self.weight = 1

    def init(self,data):
        """
        高斯分布有2种初始化方法：
         1. random  生成随机的参数，不推荐 
         2. by_data 根据分配的数据，对参数进行估计 
        """
        self.Sigma = np.cov(data,rowvar=0)
        self.mu = np.mean(data,axis=0).reshape(-1,1)

        self.inv_Sigma = la.inv(self.Sigma)
        det = np.fabs(la.det(self.Sigma))
        self.factor = (2*np.pi)**(self.dim/2)*det**0.5

    def update(self,mu,Sigma):
        self.mu = mu
        self.Sigma = Sigma
        self.inv_Sigma = la.inv(self.Sigma)
        det = np.fabs(la.det(self.Sigma))
        self.factor = (2*np.pi)**(self.dim/2)*det**0.5
        
    def cal_prob(self,x):
        """ 
        预先计算 公共部分，以减少计算量。
        x 的形状 (d,1) 
        """
        dx = x - self.mu
        # exp{(1,d)*(d,d)*(d,1)}
        numerator = np.exp(-0.5*dx.T @ self.inv_Sigma @ dx)[0,0]
        return numerator/self.factor

    def cal_probs(self,X):
        """ 
        参数: X    形状为(n,d)
        返回: Prob 形状为(n,1)
        """
        n = X.shape[0]
        Prob = np.zeros((n,1))
        for i in range(n):
            x = X[i].T.reshape(-1,1)
            prob = self.cal_prob(x)
            Prob[i] = prob
        return Prob

    def to_tuple(self):
        return self.Sigma,self.mu,self.weight

    def __str__(self):
        str_ = '--高斯分布(%.3f)--\n'%self.weight
        str_ += '均值：%s\n'%str(self.mu.T)
        str_ += '协方差：\n%s\n'%str(self.Sigma)
        return str_


class GMM:
    """
    举例：
    >>> X = np.array([[0,1],[0,2],[3,4],[4,5]])
    >>> gmm = GMM(m=2,dim=2)
    >>> gmm.fit(X,t=100)
    >>> label = gmm.which_cluster(X)
    属性值：
     dim   数据的维度
     m     高斯分布的个数
     Guass m个高斯分布
    """
    def __init__(self,m,dim):
        """
        参数：
        m   类的数目
        dim 数据的维度
        """
        self.m = m
        self.dim =dim

    def _init_parameters(self,X,method='kmeans'):
        """
        初始化高斯分布的参数。
        如果 method == 'kmeans'，那么使用kmeans进行初始化；
        如果 method == 'random'，那么进行随机初始化。
        """
        n = X.shape[0]
        self.Guass = [Guass_distribution(dim=self.dim) for i in range(self.m)]

        if method is 'kmeans':
            try:
                kmeans = Kmeans()
                labels,centroids = kmeans.main(X,k=self.m,t=100,c_strategy='kmeans')
            except:
                centroids,labels = vq.kmeans2(X,self.m,minit='points',iter=1000)
            clusters = [[j for j in range(n) if labels[j]==i] for i in range(self.m)]

        elif method is 'random':
            time_seed = int(time.time())
            np.random.seed(time_seed)   
            clusters  = [[] for i in range(self.m)]
            centroids = random.sample(list(range(n)),self.m) # 随机生成m个中心
        
            for i in range(n):
                ci = np.argmin([la.norm(X[i]-X[c]) for c in centroids])
                clusters[ci].append(i)

        else:
            raise ValueError("Unknown method type!")

        for i in range(self.m):
            guass = self.Guass[i]
            data = X[clusters[i]]
            guass.init(data)
            guass.weight = len(clusters[i])/n
        
    def _cal_prob(self,X):
        """计算每个高斯分布中，X的概率"""
        n = X.shape[0] # X (n,d)
        Prob = np.zeros((n,self.m)) # Prob (n,m)
        for i in range(self.m):
            guass = self.Guass[i]
            prob = guass.cal_probs(X)
            Prob[:,i] = prob.T * guass.weight

        denominator = np.sum(Prob,axis=1,keepdims=1)
        Prob = Prob/denominator
        return Prob

    def _estimate_parameters(self,Prob,X):
        """估计参数"""
        n = Prob.shape[0]
        Prob_sum = np.sum(Prob,axis=0) # 为了减少计算量

        for i in range(self.m):
            weight = Prob_sum[i]/n

            mu = (Prob[:,i] @ X) / Prob_sum[i] # (1,n)*(n,d)
            mu = mu.reshape(-1,1)

            Sigma = np.zeros((self.dim,self.dim))
            for t in range(n):
                xt = X[t].T.reshape(self.dim,1) # (d,1)
                Sigma += Prob[t,i] * np.outer(xt-mu,xt-mu) # (d,1)*(1,d)=(d,d)
            Sigma = Sigma/Prob_sum[i]

            self.Guass[i].weight = weight
            self.Guass[i].update(mu,Sigma)

    def _to_tuple(self):
        return [self.Guass[i].to_tuple() for i in range(self.m)]

    def _is_same(self,p1,p2):
        """判断p2和p2是否一模一样"""
        for t1,t2 in zip(p1,p2):
            for e1,e2 in zip(t1,t2):
                if (e1 != e2).any():
                    return False
        return True

    def fit(self,X,max_iter):
        """
        通过X对模型进行训练。
        参数：
         X 训练数据，形状为(n,d)
         max_iter 最大迭代次数
        """
        self._init_parameters(X) # 初始化参数
        old_parameters = self._to_tuple() # 记录上一次的参数，便于Early Stop
        for i in range(max_iter):
            if (i+1)%10 == 0:
                print ('第 %d 次迭代'%(i+1))

            # E步
            Prob = self._cal_prob(X)
            # M步
            self._estimate_parameters(Prob,X)

            # 判断模型参数是否发生变化，如果没有，则终止训练
            new_parameters = self._to_tuple()
            if self._is_same(old_parameters,new_parameters):
                print ('Early Stop, at %d'%(i+1))
                break
            else:
                old_parameters = new_parameters

        return Prob

    def which_cluster(self,X):
        """判断样本属于m个高斯分布中的哪一个"""
        Prob = self._cal_prob(X)
        labels = np.argmax(Prob,axis=1)
        return labels

    def cal_prob(self,X):
        """计算在此模型下，X存在的概率"""
        n = X.shape[0]
        Prob = np.zeros((n,self.m)) # (n,m)
        for i in range(self.m):
            guass = self.Guass[i]
            prob = guass.cal_probs(X)
            Prob[:,i] = prob.T * guass.weight

        return np.sum(Prob,axis=1)

    def show_parameter(self,precision=2):
        """展示模型参数"""
        np.set_printoptions(precision = precision)
        for guass in self.Guass:
            print (guass)


class CLF:
    """
    使用GMM对每个类中的数据分布进行建模，通过贝叶斯判别准则进行分类。
    举例：
    >>> clf = CLF(10,3,dim=17)
    >>> clf.fit(minst_train_X,minst_train_Y)
    >>> Y_predict = clf.predict(minst_test_X)
    """
    def __init__(self,k,m,dim):
        """
        参数:
         k   数据中类的数目
         m   每个GMM中高斯分布的数目
         dim 数据的维度
        """
        self.k = k
        self.gmms = [GMM(m=m,dim=dim) for i in range(k)]
        self.cluster_prob = np.array([1/k for i in range(k)]) # 在这里假设类条件概率均相等

    def fit(self,X,Y):
        """
        参数:
         X 形状为(n,d)
         Y 形状为(n,)
        """
        Y = Y.reshape((X.shape[0],))
        for i in range(self.k):
            data_index = np.nonzero(Y==i)
            data = X[data_index]
            self.gmms[i].fit(data,max_iter=200)

    def predict(self,X):
        """
        参数:
         X 形状为(n,d)
        返回:
         lable 形状为(n,)
        """
        n = X.shape[0]
        Prob = np.zeros((n,self.k))
        for i in range(self.k):
            Prob[:,i] = np.log(self.gmms[i].cal_prob(X))+np.log(self.cluster_prob.reshape((1,self.k))) # (n,k) + (1,k)
        
        label = np.argmax(Prob,axis=1)
        return label


if __name__ == '__main__':
    X = np.array([[0,0],[1,0],[0,1],[1,1],
         [2,1],[1,2],[2,2],[3,2],
         [6,6],[7,6],[8,6],[7,7],
         [8,7],[9,7],[7,8],[8,8],
         [9,8],[8,9],[9,9]])
    
    gmm = GMM(m=2,dim=2)
    gmm.fit(X,max_iter=100)
    gmm.show_parameter()
    print (gmm.cal_prob(X))