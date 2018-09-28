function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% part1

size(nn_params)
Xplus = [ones(size(X,1),1) X];  %把1加上到X矩阵中  5000*401
%Theta1 = reshape(nn_params(1:10025),25,401);  %25*401
%Theta2 = reshape(nn_params(10026:end),10,26); % 10*26

layer2 = Theta1*transpose(Xplus);  % 25*5000

layer_2 = sigmoid(layer2);  % 25*5000 

layer3 = Theta2*[ones(1,size(layer_2,2));layer_2];  % layer_2变成 26*5000
layer_3 = sigmoid(layer3); %实际上这个就是10*5000的结果了，转置下就可以得到5000*10 的表示这5000个点的分类了。而这些输出值都是在0-1

res = transpose(layer_3);  %这个结果就是5000*10的结果，也就是神经网络计算出来的结果

res(1,:);

%%%% 修改y的形式，y本来是5000*1这种格式的，但是现在要改变成5000*10的这种类型，因为要分为10类。

y0 = zeros(5000,10);
for i=1:5000
	j = y(i);
	y0(i,j) = 1;

end
% y0就是处理后的y值，是5000*10的结构

%接下来就是按照J的计算方法分别对比res和y0两个结果进行cost function计算

m = size(X, 1);

J = ((-1)/m)*sum(sum(y0.*log(res)+(1-y0).*log(1-res))) + (lambda/(2*m))*sum(nn_params.^2);























% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
