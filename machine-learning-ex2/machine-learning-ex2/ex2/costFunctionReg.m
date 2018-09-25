function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

ox = X*theta;
h = sigmoid(ox);
J = (1/m)*((-transpose(y)*log(sigmoid(ox)))-transpose(1-y)*log(1-sigmoid(ox))) + (lambda/(2*m))*sum(theta(2:end).^2);

theta_r = theta(2:end,:);
x_r = X(:,2:end);

theta_1 = theta(1,:);
x_1 = X(:,1);

% 这个里面常犯的错误是，在计算sigmoid(X*theta)是不需要去掉theta0的，因为从公式上也可以看出，只是对j进行分段，而对i则没有
grad_r = transpose((1/m)*transpose(sigmoid(X*theta)-y)*x_r) + (lambda/m)*theta_r;
grad_1 = transpose((1/m)*transpose(sigmoid(X*theta)-y)*x_1)


grad = [grad_1;grad_r];


% =============================================================

end
