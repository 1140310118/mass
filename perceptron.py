import csv
import numpy as np
import numpy.linalg as la


# Y 为样本集合，形状为(n,dim+1)
# b 为标记集合，形状为(n,1)
# a 为模型参数，形状为(dim+1,1)

class MSELoss:
    """
    均方误差：所有样本上误差平方的均值。
    """
    def __init__(self):
        pass

    def get(self,a,Y,b):
        return np.mean((Y@a-b)**2)

    def gradient(self,a,Y,b):
        n = Y.shape[0]
        return 2*Y.T@(Y@a-b)/n


class PerceptronLoss:
    """
    感知器损失：错分样本到判别界面距离之和。
    """
    def __init__(self):
        pass
    
    def get(self,a,Y,b):
        is_false_sample = (Y@a < 0).reshape((-1,1))
        return np.sum(- Y@a * is_false_sample)

    def gradient(self,a,Y,b):
        is_false_sample = (Y@a < 0).reshape((-1,1))
        return np.sum(- Y * is_false_sample,axis=0).reshape(a.shape)


class BiClassification_Perceptron:
    """
    用于二分类的感知机。
    Args:
     loss_type: 损失函数的类型，可选值
        - MSELoss (默认值)
        - PerceptronLoss
    ----------------------------------------------
    Example:
     >>> feature = np.array([[1,1],[2,2],[2,0],
                             [0,0],[1,0],[0,1]])
     >>> label = np.array([-1,-1,-1,1,1,1]).reshape((-1,1))
     >>> clf = BiClassification_Perceptron(loss_type='MSELoss')
     >>> clf.fit(feature=feature,label=label,max_iter=1000,lr=0.01)
     ## Early stop at 887
     >>> clf.predict(feature).T
     [[-1 -1 -1  1  1  1]]
     >>> clf.parameter().T
     [[-0.91349388 -0.32122789  1.12494688]]
    """
    def __init__(self,loss_type='MSELoss'):
        if loss_type=='MSELoss':
            self.loss = MSELoss()
        elif loss_type=='PerceptronLoss':
            self.loss = PerceptronLoss()
        else:
            raise ValueError('Unknown loss_type: {loss_type}'.format(loss_typp=loss_type))

    def fit(self,feature,label,max_iter,lr=0.01,therhold=1e-4):
        """
        Args:
         feature: 特征集合，形状为(n,dim)
         label: 标记集合，形状为(n,1)
         max_iter: 最大迭代次数
         lr: 学习率
         therhold: 早停止中的阈值
        """
        n,dim = feature.shape
        Y,b = self._make_Y_and_b(feature,label,n)
        self.a = np.ones((dim+1,1))
        for i in range(max_iter):
            da = self.loss.gradient(self.a,Y,b)*lr
            self.a = self.a - da
            if la.norm(da,ord=1)<therhold:
                print ('## Early stop at {step}'.format(step=i+1))
                break

    def predict(self,feature):
        n = feature.shape[0]
        Y = np.concatenate((feature,np.ones((n,1))),axis=1)
        return 2 * (Y @ self.a > 0).astype(int) - 1 # (n,dim+1)@(dim+1,1)

    def parameter(self):
        return self.a

    def _make_Y_and_b(self,feature,label,n):
        """
        添加常数列，并进行规范化
        """

        Y = np.concatenate((feature,np.ones((n,1))),axis=1)
        Y = Y*label
        b = np.ones((n,1))
        return Y,b


class MultiClassification_Perceptron:
    """
    用于多分类的感知机。
    Args:
     m: 类的数目
    -----------------------------------------------
    Example:
     >>> feature = np.array([[1,1],[2,2],[2,0],
                             [0,0],[1,0],[0,1]])
     >>> label = np.array([0,0,0,1,1,1]).reshape((-1,1))
     >>> clf = MultiClassification_Perceptron(m=2)
     >>> clf.fit(feature=feature,label=label,strategy='MSE',max_iter=1000)
     ## Early stop at 708
     >>> clf.predict(feature).T
     [0 0 0 1 1 1]
    """
    def __init__(self,m):
        self.m = m
    
    def fit(self,feature,label,max_iter,strategy='Kesler',lr=0.01,therhold=1e-4):
        """
        Args:
         strategy: 多分类的策略，可选值Kesler，MSE
         feature: 特征集合，形状为(n,dim)
         label: 标记集合，形状为(n,1)
         max_iter: 最大迭代次数
         lr: 学习率
         therhold: 早停止中的阈值
        """
        if strategy=='Kesler':
            self.fit_Kesler(feature,label,max_iter)
        elif strategy=="MSE":
            self.fit_MSE(feature,label,max_iter,lr,therhold)
        else:
            raise ValueError('Unknowd strategy {strategy}'.format(strategy=strategy))

    def fit_Kesler(self,feature,label,max_iter,batch_size=1000):
        n,dim = feature.shape
        # Y (n,dim+1), b (n,1)
        Y,b = self._make_Y_and_b(feature,label,n) 
        self.A = np.ones((dim+1,self.m)) # (dim+1,k)
        k = 0 # 样本的索引
        false_num = 0 # 早停止中使用的变量
        for i in range(int(max_iter*n/batch_size)):
            yk = Y[k:k+batch_size].reshape((dim+1,batch_size)) # (dim+1,batch_size)
            bk = b[k:k+batch_size].reshape((batch_size,)) # 样本yk的真实类别 (batch_size,)
            gs = self.A.T @ yk # m个判别函数的值 (k,dim+1)*(dim+1,batch_size)=(k,batch_size)
            
            for j in range(self.m):
                # mask 中的元素为0、1，1表示当前样本分类错误
                is_false = (gs[bk,np.arange(batch_size)]<=gs[j]).astype(int) - (bk==j).astype(int) # (batch_size,)
                for bi in range(batch_size):
                    if is_false[bi] == True:
                        self.A[:,bk[bi]] += yk[:,bi] # (dim+1,1)
                        self.A[:,j]      -= yk[:,bi] # (dim+1,1) 
                        false_num += 1
            # Early stop
            if k == n-1:
                if (i+1)%(10*n) == 0:
                    print ((i+1)/n)
                if false_num == 0:
                    print ('## Early stop at {step}'.format(step=i+1))
                    break
                false_num = 0
            k = (k+batch_size)%n

    def fit_MSE(self,feature,label,max_iter,lr=0.01,therhold=1e-4):
        loss = MSELoss()
        n,dim = feature.shape
        Y,b = self._make_Y_and_b(feature,label,n,onehot=True)
        self.A = np.ones((dim+1,self.m))
        print (Y.shape,b.shape,self.A.shape)
        for i in range(max_iter):
            dA = loss.gradient(self.A,Y,b)*lr/((i+1)//100 +1)
            self.A = self.A - dA
            if la.norm(dA,ord=1)<therhold*self.m:
                print ('## Early stop at {step}'.format(step=i+1))
                break
            # print (loss.get(self.A,Y,b))

    def predict(self,feature):
        n = feature.shape[0]
        Y = np.concatenate((feature,np.ones((n,1))),axis=1)
        B = Y@self.A # (n,dim+1) (dim+1,k)
        return np.argmax(B,axis=1).reshape((-1,1))

    def _make_Y_and_b(self,feature,label,n,onehot=False):
        """
        添加常数列
        """
        Y = np.concatenate((feature,np.ones((n,1))),axis=1) # (n,dim+1)
        if onehot == False:
            b = label # (n,1)
        else:
            b = (np.arange(self.m)==label).astype(int) # (n,m)
            
        return Y,b

def load_data(filename,type_=float):
	D = []
	with open(filename,'r',encoding='utf-8') as file_in:
		reader = csv.reader(file_in)
		for row in reader:
			d = [type_(i) for i in row]
			D.append(d)
	return np.array(D)

def transform(arr):
    processor = MinMaxScaler()
    processor.fit(arr)
    return processor.transform(arr)


if __name__=="__main__":
    from sklearn.preprocessing import MinMaxScaler



    feature = load_data('data/TrainSamples.csv')[:30000]
    feature = transform(feature)
    label   = load_data('data/TrainLabels.csv',type_=int)[:30000]
    # feature = np.array([[1,1],[2,2],[2,0],
                            #  [0,0],[1,0],[0,1]])
    # label = np.array([0,0,0,1,1,1]).reshape((-1,1))
    


    clf = MultiClassification_Perceptron(m=10)
    clf.fit(feature=feature,label=label,strategy='Kesler',max_iter=100,lr=1e-1,therhold=1e-5)
    label_predict = clf.predict(feature)

    print (label_predict.T)
    print (label.T)
    print (np.sum(label_predict==label))
    # print (clf.parameter().T)