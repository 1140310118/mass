"""
参考：模式识别课件/第8章 成分分析与核函数
"""

import numpy as np


class FisherDiscriminantAnalysis:
    def __init__(self):
        self.k = 1
        self.zero = 1e-5

    def fit(self, X, y):
        Sw, Sb = self._cal_Sw_Sb(X, y)
        self.W = self._positive_eigenvectors(np.linalg.inv(Sw) @ Sb)

    def transform(self, X):
        return X @ self.W

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def _cal_Sw_Sb(self, X, y):
        """
        计算类内散度矩阵Sw及类间散度矩阵Sb
        """
        global_mean = np.mean(X, axis=0)
        Sw = Sb = 0
        for label_type in set(y):
            index = (y == label_type)
            mean = np.mean(X[index], axis=0)
            dx = X - mean
            Sw += dx.T @ dx
            dm = (mean - global_mean).reshape((-1, 1))
            Sb += dm @ dm.T * len(index)
        return Sw, Sb

    def _positive_eigenvectors(self, matrix):
        """
        :param matrix: 必须是一个方阵
        :return: 返回特征值大于0的特征向量
        """
        values, vectors = np.linalg.eig(matrix)
        indexes = (values>self.zero)
        return vectors[:, indexes]


if __name__ == '__main__':
    X = np.array([[-5, -4], [-4, -5], [-5, -6], [-6, -5],
                  [5, 4], [4, 5], [5, 6], [6, 5]])

    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    pca = FisherDiscriminantAnalysis()
    X_transformed = pca.fit_transform(X, y)
    print(X_transformed)
