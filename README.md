# mass
代码小站，杂杂的东西

## 聚类算法

- Kmeans
- GMM

## 分类算法

- 感知器Perceptron

$$
if\ y_k\in w_i {\rm\ and\ }g_i(y_k)\leq g_j(y_k)\\
then\ a_i += y_k,a_j -=y_k.
$$

```
if g[target]<=g[j]:
	a[target] += yk
	a[j]      -= yk
即 a += yk*mask, 其中
mask[i]=-1,if g[target]<=g[i];
mask[i]=1,if i=target and g[target]<=g[i]
mask[i]=0,if 
```



## 字符串算法

- AC自动机 ——多模式匹配，简介 [AC算法及python实现](http://superzhang.site/blog/AC-algorithm-and-its-python-implementation/)
