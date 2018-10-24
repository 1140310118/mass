"""
2018/10/24
参考：模式识别课件/成分分析与核函数
"""

import numpy as np


class PrincipalComponentAnalysis:
    def __init__(self,k=1):
        self.k = k

    def fit(self, X):
        sigma = np.cov(X, rowvar=False)
        self.mu = np.mean(X, axis=0)
        self.E = self._topk_eigenvectors(sigma)

    def transform(self, X):
        return  (X-self.mu) @ self.E

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def _topk_eigenvectors(self, matrix):
        """
        :param matrix: 必须是一个方阵
        :return: 返回前k个特征向量
        """
        values, vectors = np.linalg.eig(matrix)
        indexes = np.argsort(values)
        topk_indexes = indexes[:-self.k-1:-1]
        return vectors[:,topk_indexes]


if __name__ == '__main__':
    X = np.array([[-5, -4], [-4, -5], [-5, -6], [-6, -5],
                  [5, 4], [4, 5], [5, 6], [6, 5]])

    pca = PrincipalComponentAnalysis(k=1)
    X_transformed = pca.fit_transform(X)
    print(X_transformed)
