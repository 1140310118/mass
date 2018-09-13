import numpy as np 
import random


class Kmeans:
	"""
	举例：
	>>> X = np.array([[0,1],[0,2],[3,4],[4,5]])
	>>> Kmeans = Kmeans()
	>>> Kmeans.main(X,k=2,t=10)
	[1, 1, 0, 0]
	"""
	def __init__(self):
		pass

	def cal_distance(self,p1,p2,dim=2):
		"""
		计算p1,p2之间的欧式距离，p1、p2的形状为(*,*,d)。
		若要使用其他的距离度量方式，可以重写此函数。
		"""
		return np.sum((p1-p2)**2,dim)

	def _cal_internal_distance(self,p):
		"""
		计算p中样本间的距离，p的形状为(m,d)，返回一个方阵。
		"""
		m = p.shape[0]
		D = np.zeros((m,m))
		for i in range(m):
			D[:,i] = self.cal_distance(p,p[i],dim=1)
		return D

	def _init_centers(self,X,n,k):
		"""
		从n个样本中，随机选取k个样本，作为类中心。
		"""
		center_index = random.sample(list(range(n)),k)
		centers = X[center_index]
		return centers

	def _classify(self,X,centers,k,n):
		"""
		X (n,d)
		"""
		centers = centers.reshape(k,1,-1) # (k,1,d)
		distances = self.cal_distance(X.reshape(1,n,-1),centers) # (1,n,d)-(k,1,d)->(k,n,d) 计算所有样本与所有中心的距离
		labels = np.argmin(distances,axis=0) # 选择距离最小的类中心，作为自己的类别
		return labels

	def _get_centers(self,X,labels,k,strategy='kmeans'):
		"""
		为每个类，重新计算类的中心。
		Args:
			strategy: kmeans, 将类内距其他样本距离最小的点作为新的中心；
					  kme（忘记全称是什么了）直接将类内所有样本的均值作为新的中心，计算量相对较少。
		"""
		centers = []
		for i in range(k):
			
			iclass_index = np.nonzero(labels == i) # 第i个类中所有样本的下标
			iclass = X[iclass_index]
			if strategy == 'kmeans':
				D = self._cal_internal_distance(iclass) # 样本间的距离矩阵 方阵
				ds = np.sum(D,axis=0) # 一个类内的样本与其他样本的距离之和
				center = iclass[np.argmin(ds)]
			elif strategy == 'kme':
				center = np.mean(iclass,axis=0)
			centers.append(center)

		return np.array(centers)


	def main(self,X,k,c_strategy='kmeans',t=10):
		"""
		针对数值型数据，使用Kmeans方法进行聚类，返回每个样本对应的类别

		Args:
			X 待聚类的数据 形状(n,d)
			k 类的数目
			t 迭代次数
			c_strategy 计算类中心的策略，kmeans、kme

		Returns:
			labels: 样本对应的类别
			centers: 每个类中心点的下标
		"""
		n,d = X.shape
		centers = self._init_centers(X,n,k) # 初始化类的中心
		for i in range(t):
			print ('第 %d 次迭代'%i)
			labels = self._classify(X,centers,k,n) # 根据类的中心对样本进行分类
			old_centers = centers[:]
			centers = self._get_centers(X,labels,k,c_strategy) # 重新计算类中心
			if (centers == old_centers).all(): # 对比类中心是否变化，如没有则退出
				print('early stop, 迭代次数 %d'%(i+1))
				break
		return labels,centers
			


if __name__ == "__main__":

	X = np.array([[0,0],[1,0],[0,1],[1,1],
		 [2,1],[1,2],[2,2],[3,2],
		 [6,6],[7,6],[8,6],[7,7],
		 [8,7],[9,7],[7,8],[8,8],
		 [9,8],[8,9],[9,9]])

	Kmeans = Kmeans()
	l,c = Kmeans.main(X,k=2,t=10,c_strategy='kmeans')
	print ("类标签：",*l)
	print ("类中心：",*c)