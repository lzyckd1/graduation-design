# ICP算法

该算法旨在解决这样一个问题：给出一个点集A，以及其经过姿态变化(包括一个旋转变化R和平移变化T)后的点集B，将他们之间的点两两对应并反推出两个变换矩阵，更进一步，考虑原物体被抓取的点云和变换后抓取到的点云并不是点与点之间完全对应，如何找到相应的变换矩阵使点云A变换后与B的差距最小。

一个自然的想法是，如果点云之间的变换没有平移，处理起来会简单得多。我们先求出两个点云各自的质心，然后将各点的坐标与质心坐标相减，这样得到的A'和B'两个点云相当于是以原点为质心，这时我们只需要考虑A'关于原点旋转到B'的旋转矩阵即可。

接下来求旋转矩阵。

设A中的各点为$p_i$，B中各点为$q_i$，loss为$\sum{^N_{i=1}}||Rp_i-q_i||^2$

化简此式，有：
$$
||Rp_i-q_i||^2=(Rp_i-q_i)^T(Rp_i-q_i)=p_i^TR^TRp_i+q_i^Tq_i-2p_i^TRq_i
$$
又旋转矩阵为正交矩阵，故有$R^TR=I$，要使loss最小，就要使$-2p_i^TRq_i$最小，即有
$$
R^*=argmax (\sum p_i^TRq_i)=argmax(trace(P^TRQ))
$$
而$trace(AB)=trace(BA)$，故有
$$
R^*=argmax(trace(RQP^T))
$$
令$H=QP^T=U\Sigma V^T$(奇异值分解)，有
$$
trace(RQP^T)=trace(RH)=trace(RU\Sigma V^T)=trace(\Sigma V^TRU)
$$
而$VRU$均为正交矩阵，奇异值矩阵中元素值非负，且在其类似于主对角线那条线之外的元素值均为0，可以得到结论：$V^TRU=I$时值最大，故$R^*=VU^T$。

此外，此时求得的$R^*$不一定是旋转矩阵，也有可能是映射矩阵(有手性变化)，根据数学上的推导，
$$
R^*=V\left[
 \begin{matrix}
   1 & 0 & 0 \\
   0 & 1 & 0 \\
   0 & 0 & |VU^T|
  \end{matrix}
  \right] U^T
$$
下面是基于源码的一些解释：

* 下面的函数是找到对于src中的每个点，找到其在dst中对应的最近点

  ```
  def nearest_neighbor(src, dst):
      indecies = np.zeros(src.shape[0], dtype=np.int)
      distances = np.zeros(src.shape[0])
      for i, s in enumerate(src):
          min_dist = np.inf
          for j, d in enumerate(dst):
              dist = np.linalg.norm(s - d)
              # 求范数
              if dist < min_dist:
                  min_dist = dist
                  indecies[i] = j
                  distances[i] = dist
      return distances, indecies
  ```

​	这段代码以下问题：

   	1. 复杂度过高
   	2. 可能会出现src中多个点被映射到同一个dst的点中

​    就复杂度高这一点上，可以使用KDTree或者近似最近邻搜索来优化

* 下面的函数为

  ```
  def best_fit_transform(A, B):
      assert len(A) == len(B)
  
      # 将原点云变换为平移至质心与原点重合的新点云
      centroid_A = np.mean(A, axis=0)
      centroid_B = np.mean(B, axis=0)
      AA = A - centroid_A
      BB = B - centroid_B
  
      # 计算旋转矩阵，具体数学推动见前
      W = np.dot(BB.T, AA)
      U, s, VT = np.linalg.svd(W)
      R = np.dot(U, VT)
  
      # 即前面提到的R可能并不是旋转矩阵而是映射矩阵
      if np.linalg.det(R) < 0:
          VT[2, :] *= -1
          R = np.dot(U, VT)
  
      # 偏差值
      t = centroid_B.T - np.dot(R, centroid_A.T)
  
      # homogeneous transformation
      T = np.identity(4)
      T[0:3, 0:3] = R
      T[0:3, 3] = t
  
      return T, R, t
  ```

* 下面函数为迭代部分

  ```
  def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
      '''
      The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
      Input:
          A: Nxm numpy array of source mD points
          B: Nxm numpy array of destination mD point
          init_pose: (m+1)x(m+1) homogeneous transformation
          max_iterations: exit algorithm after max_iterations
          tolerance: convergence criteria
      Output:
          T: final homogeneous transformation that maps A on to B
          distances: Euclidean distances (errors) of the nearest neighbor
          i: number of iterations to converge
      '''
  
      assert A.shape == B.shape
  
      # get number of dimensions
      m = A.shape[1]
  
      # make points homogeneous, copy them to maintain the originals
      src = np.ones((m+1,A.shape[0]))
      dst = np.ones((m+1,B.shape[0]))
      src[:m,:] = np.copy(A.T)
      dst[:m,:] = np.copy(B.T)
  
      # init_pose是一个初始姿态变换
      if init_pose is not None:
          src = np.dot(init_pose, src)
  
      prev_error = 0
  
      for i in range(max_iterations):
          # 找到当前src对应到dst中的最近点，得到对应关系
          distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)
  
          # 计算对应关系所需要的旋转矩阵以及偏差值
          T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)
  
          # 更新src当前矩阵
          src = np.dot(T, src)
  
          # check error
          mean_error = np.mean(distances)
          if np.abs(prev_error - mean_error) < tolerance:
              break
          prev_error = mean_error
  
      # 计算最终的变换矩阵，此时的src是原A经过变换后最接近B的点云
      T,_,_ = best_fit_transform(A, src[:m,:].T)
  
      return T, distances, i
  ```

  值得注意的一点是在本段代码中，src为4*N而非3*N，其前3行为点，第4行为每次经矩阵变换后的偏差值，这一点尚有疑问，似乎在数学推导中没有直接体现这一步。



# ICP的改进之一——KDTree&ANN

即基于KDTree的近似最近邻搜索

KDTree的大致思想类似于二叉搜索树：

* 节点的域：
  * node data：数据矢量
  * range：改节点所代表的空间范围
  * split：一个整数，垂直于分割超平面的方向轴序号
  * left：超平面左子空间构建的KDTree
  * right
  * parent

* 建树：类似于二叉树
  * 方法一，以各维度分别为第一、第二关键字，以此类推，这样就能够比较点的大小，从而划分空间
  * 方法二，第i次分割分别以i%n维为关键字进行点的比较。
  * 方法三，考虑在各个维度上各点的数据方差，选择方差中最大的值，对应维为split域的值。
  * ...
* 搜索，就和二叉树类似，通过建树时的点的比较方法在书上找到一条搜索路径，在路径上找到最近点

在此种算法下，对于规模为N的点集，虽然不能保证每个点最终找到最近点，但是时间复杂度由原先的$o(N^2)$下降到了$o(NlogN)$。

# ICP算法的其他改进

## 基于不同loss的变种

* point to plane，考虑的不再是点到对应的最近点，而是点到一个相应的平面的最近距离。

  $loss=\sum((R*p_i+t-q_i)*n_i)$，n为对应平面的法向量。但其优化是一个非线性问题，速度较慢，可以在一定假设下近似为线性问题求解，方法与point to point一致，最后表达式依然是基于SVD求解，可以参考[这里](https://www.jianshu.com/p/57eecfa9ca6f?utm_campaign=maleskine&utm_content=note&utm_medium=seo_notes&utm_source=recommendation)。

* plane to plane，只考虑原点云与目标点云局部结构

* generalized ICP，综合了point to point，point to plane和plane to plane

* normal ICP，考虑了法向量和局部向量

## 基于实际应用的一些变种

* colored ICP，针对RGB点云，是基于point to plane上的进一步改进，这个算法可能比较重要
* Symmetric ICP，对于point to plane的改进，将$n_i$改为$n_{pi}+n_{qi}$
