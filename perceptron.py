import csv
import math
import numpy as np 
import numpy.linalg as la 
from sklearn.preprocessing import MinMaxScaler

# Y 为样本集合，形状为(n,dim+1)
# b 为标记集合，形状为(n,1)
# a 为权值矢量，形状为(dim+1,1)


class BiClassification_Perceptron:
    '''
    用于二分类的感知器。
    --------------------------------------------
    Example:
     >>> feature = np.array([[1,1],[2,2],[2,0],
                             [0,0],[1,0],[0,1]])
     >>> label = np.array([-1,-1,-1,1,1,1]).reshape((-1,1))
     >>> clf = Biclassification_Perceptron()
     >>> clf.fit(feature,label)
     Early stop at iter_num 54
     >>> clf.preidct(feature).T 
     [[-1 -1 -1  1  1  1]]
     >>> clf.parameter().T
     [[-0.25 -0.02  0.27]]
    '''
    def __init__(self):
        pass

    def fit(self,feature,label,max_iter=200,lr=0.01,
                 therhold=1e-4,loss_type='Perceptron',para_init='uniform'):
        """
        Args:
         feature: 特征矩阵，形状为(n,dim)
         label: 标记向量，形状为(n,1)
         max_iter: 最大迭代次数
         lr: 学习率
         therhold: 用于早停止的阈值
         loss_type: 损失函数的类型
            - MSE (默认值)
            - Perceptron
         para_init: 参数初始化方式
            - uniform (默认值): 初始化为1
            - random: 随机初始化
        """
        
        dim = feature.shape[1]

        if para_init == 'uniform':
            self.a = np.ones(shape=(dim+1,1)) 
        elif para_init == 'random':
            self.a = np.random.random(size=(dim+1,1))
        else:
            raise ValueError('Unknow parameter initialize type: {para_init_type}'.format(
                para_init_type = para_init ))

        self._fit(feature,label,max_iter,lr,therhold,loss_type)
            
    
    def _fit(self,feature,label,max_iter,lr,therhold,loss_type):
 
        n = feature.shape[0]
        Y,b = self._make_Y_and_b(feature,label,n)
        for i in range(max_iter):
            is_false = (Y@self.a<=0).reshape((n,1))
            if loss_type == 'Perceptron':
                gradient = np.sum(-Y*is_false,axis=0).reshape(self.a.shape)
            elif loss_type == 'MSE':
                gradient = 2*Y.T@(Y@self.a-b)/n 
            else:
                raise ValueError('Unknow Loss Type: {loss_type}'.format(
                    loss_type = loss_type ))
            da = - lr*gradient

            self.a = self.a + da
            if la.norm(da,ord=1)<therhold:
                print ('Early stop at iter_num {iter_num}'.format(iter_num=i+1))
                return

    def predict(self,feature):
        Y = np.concatenate((feature,np.ones((feature.shape[0],1))),axis=1)
        return 2*(Y@self.a>0)-1
    
    def parameter(self):
        return self.a
    
    def _make_Y_and_b(self,feature,label,n):
        Y = np.concatenate((feature,np.ones((n,1))),axis=1)
        Y = Y*label
        return Y,np.ones_like(label)


def get_batch(Y,b,batch_size,shuffle=True,label_width=1):
    n = Y.shape[0]
    batch_num = math.ceil(n/batch_size)
    if shuffle == True:
        Yb = np.concatenate((Y,b),axis=1)
        np.random.shuffle(Yb)
        Y = Yb[:,:-label_width]
        b = Yb[:,-label_width:].astype(int)
    for i in range(batch_num):
        yield Y[i*batch_size:(i+1)*batch_size],b[i*batch_size:(i+1)*batch_size]


# Y 为样本集合，形状为(n,dim+1)
# b 为标记集合，形状为(n,1)或(n,k)
# A 为权值矩阵。形状为(k,dim+1)

class MSELoss:
    """
    Args:
     act_func: 激活函数的类型
      - None (默认值): 不适用激活函数
      - tanh
      - sigmoid
      - relu
    """
    def __init__(self,act_func=None):
        print (act_func)
        self.act_func = act_func

    def get(self,A,Y,b):
        if self.act_func is None:
            loss = np.mean((Y@A.T-b)**2)
        elif self.act_func == 'tanh':
            z = Y@A.T
            fz = self._tanh(z)
            loss = np.mean((fz-b)**2)
        elif self.act_func == 'sigmoid':
            z = Y@A.T
            fz = self._sigmoid(z)
            loss = np.mean((fz-b)**2)
        elif self.act_func == 'relu':
            z = Y@A.T
            fz = self._relu(z)
            loss = np.mean((fz-b)**2)
        else:
            raise ValueError('Unknow Activation Function {act_func}'.format(
                act_func=self.act_func))
        return loss

    def gradient(self,A,Y,b):
        # f(z) = sigmoid(z)
        # f(z)' = f(z)(1 − f(z))
        # f(z) = tanh(z)
        # f(z)' = 1 − f(z)^2
        n = Y.shape[0]
        if self.act_func is None:
            gradient_ = (Y@A.T-b).T@Y*2/n
        elif self.act_func == 'tanh':
            z = Y@A.T
            fz = self._tanh(z)
            gradient_ = ((fz-b)*(1-fz**2)).T@Y*2/n
        elif self.act_func == 'sigmoid':
            z = Y@A.T
            fz = self._sigmoid(z)
            gradient_ = ((fz-b)*(1-fz)*fz).T@Y*2/n
        elif self.act_func == 'relu':
            z = Y@A.T
            fz = self._relu(z)
            fz_ = (fz>=0).astype(int)
            gradient_ = ((fz-b)*fz_).T@Y*2/n
        else:
            raise ValueError('Unknow Activation Function {act_func}'.format(
                act_func=self.act_func))
        return gradient_
    
    def _sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    def _tanh(self,x):
        return np.tanh(x)

    def _relu(self,x):
        return np.max((x,np.zeros_like(x)),axis=0)


class MultiClassification_Perceptron:
    '''
    用于多分类的感知器。
    -----------------------------------
    Example:
     >>> # load feature and label
     >>> perceptron_para = {
            'loss_type':'Perceptron',
            'para_init':'random'
         }
     >>> clf = MultiClassification_Perceptron(k=10)
     >>> clf.fit(feature_train,label_train,max_iter=200,
                 batch_size=1000,**perceptron_para)
     >>> clf.score(feature_test,label_test)
    '''
    def __init__(self,k):
        self.k = k
        self.act_func = None
    
    def fit(self,feature,label,max_iter,lr=0.1,therhold=1e-4,
                 loss_type='Perceptron',para_init='uniform',
                 batch_size=100,act_func='tanh',scheduler=None,shuffle=True):
        """
        Args:
         feature: 特征矩阵，形状为(n,dim)
         label: 标记向量，形状为(n,1)
         max_iter: 最大迭代次数
         lr: 学习率
         scheduler: 当损失函数为MSE时用于调整学习率，默认
            值 lambda iter_num: 0.5 ** (iter_num // 30)
         batch_size: 批的大小
         shuffle: 将数据打包成批的时候，是否随机重排
         therhold: 当损失函数为Perceptron时，用于早停止的阈值
         loss_type: 损失函数的类型
            - MSE (默认值)
            - Perceptron
         para_init: 参数初始化方式
            - uniform (默认值): 初始化为1
            - random: 随机初始化
         act_func: 激活函数的类型
            - None (默认值): 不使用激活函数
            - tanh
            - sigmoid
            - relu
        """
        dim = feature.shape[1]

        if para_init == 'uniform':
            self.A = np.ones(shape=(self.k,dim+1)) 
        elif para_init == 'random':
            self.A = np.random.random(size=(self.k,dim+1))
        else:
            raise ValueError('Unknow parameter initialize type: {para_init_type}'.format(
                para_init_type = para_init ))

        if loss_type == 'Perceptron':
            self.act_func = None
            self._fit_Kesler(feature,label,max_iter,batch_size)
        elif loss_type == 'MSE':
            self.act_func = act_func
            self._fit_MSE(feature,label,max_iter,lr,therhold,batch_size,scheduler,shuffle)
        else:
            raise ValueError('Unknow Loss Type: {loss_type}'.format(
                    loss_type = loss_type ))
    
    def _fit_Kesler(self,feature,label,max_iter,batch_size,patience=50,therhold=10):
        Y,b = self._make_Y_and_b(feature,label)
        # 用于Early stop的参数
        min_false_num = Y.shape[0]
        patient_iter_num = 0

        for iter_num in range(max_iter):
            false_num = 0
            batch_size = max(int(batch_size*0.99),100)
            for batch_Y,batch_b in get_batch(Y,b,batch_size):
                batch_size_real = batch_b.shape[0]
                target = (batch_b.reshape((-1,)),np.arange(batch_size_real))
                g = self.A@batch_Y.T

                Mask = - (g>=g[target]).astype(int)
                Mask[target] = 0  
                false_num += - np.sum(Mask)
                Mask[target] = - np.sum(Mask,axis=0)
                lr = 10/(batch_size) # lr*0.1**(iter_num//100)
                self.A = self.A + Mask@batch_Y*lr

            if iter_num%10 == 0:
                print ('Iter_num: {iter_num:3d}, Batch_size: {batch_size:4d}, False_num: {false_num:6d}, Min_false_num: {min_false_num:6d}'.format(
                        iter_num=iter_num,false_num=false_num,
                        batch_size=batch_size,min_false_num=min_false_num))
            # Early stopping 
            if false_num == 0: # 当样本线性可分时，有可能达到此条件
                print ('Early stop at iter_num {iter_num}'.format(iter_num=iter_num+1))
                return 
            else:
                if false_num < min_false_num - therhold: # 性能有所改善，则重置 patient_iter_num
                    patient_iter_num = 0
                else:
                    patient_iter_num += 1
                    if patient_iter_num >= patience:
                        print ('Early stop at iter_num {iter_num}'.format(iter_num=iter_num+1))
                        break
            min_false_num = min(min_false_num,false_num) 

    def _fit_MSE(self,feature,label,max_iter,lr,therhold,batch_size,scheduler,shuffle):
        if scheduler is None:
            scheduler = lambda iter_num: 0.5 ** (iter_num // 30)
        Y,b = self._make_Y_and_b(feature,label,onehot=True)
        self.loss = MSELoss(self.act_func)
        for iter_num in range(max_iter):
            loss_sum = 0
            _lr = max(lr * scheduler(iter_num),1e-5)
            for batch_Y,batch_b in get_batch(Y,b,batch_size,shuffle=shuffle,
                                             label_width=self.k):
                dA = - self.loss.gradient(self.A,batch_Y,batch_b)*_lr
                self.A = self.A + dA
                loss_sum += self.loss.get(self.A,batch_Y,batch_b)
            if iter_num%10 == 0:
                print ("Iter_num {iter_num:3d}, loss {loss:.3f}".format(
                        iter_num=iter_num, loss=loss_sum))
    
    def predict(self,feature):
        n = feature.shape[0]
        Y = np.concatenate((feature,np.ones((n,1))),axis=1)

        if self.act_func is None:
            onehot_predict = self.A@Y.T
        elif self.act_func == 'tanh':
            onehot_predict = self.loss._tanh(self.A@Y.T)
        elif self.act_func == 'sigmoid':
            onehot_predict = self.loss._sigmoid(self.A@Y.T)
        elif self.act_func == 'relu':
            onehot_predict = self.loss._relu(self.A@Y.T)
        else:
            raise ValueError('Unknow Activation Function {act_func}'.format(
                act_func=self.act_func))

        label_predict = np.argmax(onehot_predict,axis=0).reshape((-1,1))
        return label_predict
    
    def score(self,feature,label):
        label_predict = self.predict(feature)
        return np.sum(label_predict == label)/feature.shape[0]
    
    def parameter(self):
        return self.A

    def _make_Y_and_b(self,feature,label,onehot=False):
        n = feature.shape[0]
        Y = np.concatenate((feature,np.ones((n,1))),axis=1)
        if onehot == True:
            b = (np.arange(self.k)==label).astype(int)
        else:
            b = label
        return Y,b


def load_csv(filename,type_=float):
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

def load_data(feature_file,label_file):
    feature = load_csv(feature_file)
    feature = transform(feature)
    label   = load_csv(label_file,type_=int)
    return feature,label

def BiClassification_Perceptron_test():
    feature = np.array([[1,1],[2,2],[2,0],
                             [0,0],[1,0],[0,1]])
    label = np.array([-1,-1,-1,1,1,1]).reshape((-1,1))
    clf = BiClassification_Perceptron()
    clf.fit(feature,label)
    print (clf.predict(feature).T)
    print (clf.parameter().T)

def MultiClassification_Perceptron_test():
    
    perceptron_para = { # 0.8237
        'loss_type':'Perceptron',
        'para_init':'random'
    }  
    naive_mse_para = { # 0.7848
        'loss_type': 'MSE',
        'act_func': None,
        'para_init': 'random',
        'scheduler': lambda iter_num: 0.9 ** (iter_num//20),
        'lr':1e-1,
    }
    tanh_para = { # 0.782
        'lr':1,
        'act_func': 'tanh',
        'loss_type':'MSE',
        'scheduler': lambda iter_num: 0.8 ** (iter_num//20),
        'para_init': 'random'
    }
    sigmoid_para = { # 0.8349
        'lr':6.5,
        'act_func':'sigmoid',
        'loss_type':'MSE',
        'scheduler': lambda iter_num: 0.98 ** (iter_num//10),
        'para_init': 'random'
    }
    relu_para = { # 0.8384
        'lr':3,
        'act_func':'relu',
        'loss_type':'MSE',
        'scheduler': lambda iter_num: 0.7 ** (iter_num//10),
        'para_init': 'random',
        'shuffle': True,
    }
    feature_train,label_train = load_data('data/TrainSamples.csv','data/TrainLabels.csv')
    feature_test, label_test  = load_data('data/TestSamples.csv','data/TestLabels.csv')
    
    clf = MultiClassification_Perceptron(k=10)
    for para in (perceptron_para,naive_mse_para,
                tanh_para,sigmoid_para,relu_para):
        clf.fit(label_train,label_train,max_iter=200,
                batch_size=1000,**para)
        print (clf.score(feature_train,label_train))
        print (clf.score(feature_test,label_test))

if __name__=='__main__':
    BiClassification_Perceptron_test()
    MultiClassification_Perceptron_test()