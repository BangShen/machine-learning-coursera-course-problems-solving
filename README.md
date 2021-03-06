
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
## Coursera-Machine-Learning-Andrew-NG
This is a repository of my coursera Machine Learning by Standford, Andrew NG course's assignments' solutions

This is all written in Octave. The datasets are also uploaded with the excercises

### Includes:
* Linear Regression (one variable and multiple variables)
* Logistic Regression
* Regularization
* Neural networks
* Support Vector Machine
* Clustering
* Dimensionality Reduction
* Anomaly Detection
* Recommender System
For more information, please visit: https://www.coursera.org/learn/machine-learning


### Notes


#### Ex4
* **关于在submit的时候结果和run不一样**:这个是由于我在run的时候的代码是写死了的，但是在submit的时候的变量改变了。

* 计算正则后的J的时候，需要注意的是要去掉x0所对应的参数值，这些参数是不需要进行正则化的。这在[文章](https://stats.stackexchange.com/questions/86991/reason-for-not-shrinking-the-bias-intercept-term-in-regression)中有介绍的，对LR和neural working来说，**改变theta0就是在上下移动模拟曲线的位置，这解决不了过拟合的问题，也就是违背了正则化的目的，所以是不需要的**
。但是在其他的一些模型中，是否正则偏差项好像影响不大。

* 为了避免初始化参数容易导致的对称性问题，需要随机初始化的参数值，其方法就是$$\Theta = rand(10,11) * 2 * \epsilon - \epsilon$$， 但是问题是如何确定\\(\epsilon\\)呢，在Andrew的课程练习中提到用的方法是\\(\epsilon<sub>init</sub> = \cfrac{\sqrt{6}}{\sqrt{L_in + L_out}} \\).


* 关于backpropagation可以看这个[blog](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)中的介绍，通过数值的介绍能够更加清楚，这个里面介绍怎么实现具体的gradient计算的(发现在学习的时候能够加上numerial example
，或者step by step 这种关键词能够找到比较好的解释案例)


* <font color="blue">你在这个过程中学到哪些？</font>
	1. 是一步一步解决问题的方法就如backprogapation一样，乍一看总是觉觉得好难，但是通过课程中不断地分解步骤来解决问题，从而最终解决掉。
	2. 先对每一个样本进行单独训练，所以我这里面对5000\*400的X加了个循环，这样问题简单一些 
	3. 代码最后的部分并没有解决如何处理的，这一部分我是直接参考了别人的code.

#### Ex5
**关于练习中J<sub>train</sub>和J<sub>cv</sub>的思考**</br>

在计算J<sub>train</sub>的时候，所用的参数theta是在不同样本数量下得到的，而J<sub>train</sub>也是在该样本数量下对应的theta计算得到的。而对于J<sub>cv</sub>其不同的是，其计算样本是固定的，只是参数变化了.
可以参见下面的代码，就是在做这样的事情，error_train得到的J<sub>train</sub>,而J<sub>cv</sub>是放在了error_val中.

```matlab
for i = 1:m
	[theta] = trainLinearReg(X(1:i,:), y(1:i), lambda);
	theta
	% fprintf ("\t%d of \t %d", i,theta);
	error_train(i) = linearRegCostFunction(X(1:i,:), y(1:i), theta, 0);
	error_train;
	error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);
end
```


#### Ex7
pca 这一块也不是特别懂，codes多是参考自github中


注意点:
1. 在画学习曲线的时候的cost function不需要加正则项，这是由于我们在计算J<sub>train</sub>和J<sub>cv</sub>的时候需要的是真正的cost, 而不是正则后的，这个问题在[stack](https://stats.stackexchange.com/questions/222493/why-do-we-use-the-unregularized-cost-to-plot-a-learning-curve)上已经被讨论过


#### Ex8

* 作业中关于collaborative filtering learning的相关内容比较难一些
* 在cofiCostFunc中关于`theta_grad`和`x_grad`的计算实际上上是参考了[link](https://github.com/AvaisP/machine-learning-programming-assignments-coursera-andrew-ng/blob/master/machine-learning-ex8/ex8/cofiCostFunc.m)代码，但是自己写了很久都没转过来是怎么回事。其计算的公式可以见pdf说明中的2.2.2。



