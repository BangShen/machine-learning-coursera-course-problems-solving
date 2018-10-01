
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


##### Ex5
**关于联系中J<sub>train</sub>和J<sub>cv</sub>的思考**
在计算J<sub>train</sub>的时候，所用的参数theta是在不同样本数量下得到的，而J<sub>train</sub>也是在该样本数量下对应的theta计算得到的。而对于J<sub>cv</sub>其不同的是，其计算样本是固定的，只是参数变化了

'''matlab
for i = 1:m
	[theta] = trainLinearReg(X(1:i,:), y(1:i), lambda);
	theta
	% fprintf ("\t%d of \t %d", i,theta);
	error_train(i) = linearRegCostFunction(X(1:i,:), y(1:i), theta, 0);
	error_train;
	error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);
end

'''