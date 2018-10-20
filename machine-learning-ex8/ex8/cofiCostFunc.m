function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_featupredict, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_featupredict, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_featupredict), num_movies, num_featupredict);
Theta = reshape(params(num_movies*num_featupredict+1:end), ...
                num_users, num_featupredict);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_featupredict matrix of movie featupredict
%        Theta - num_users  x num_featupredict matrix of user featupredict
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_featupredict matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_featupredict matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

YR = Y.*R;  % Y表示的是评分矩阵，R表示的是否进行了评分


predict = transpose(Theta*transpose(X));   %这个计算的就是theta和x乘积之后得到的预测评分
size(predict);


% 下面用的方法是非vectoried的，这种方法很繁琐，最下面的计算用的就是矩阵方法
%{  
sum = 0; 		
for i = 1:size(predict,1)
	for j = 1:size(predict,2)
		if R(i,j) ~= 0
			sum = sum + (predict(i,j)-Y(i,j))^2;
		end
	end

end


J = (1/2) * sum;
%}


% 这是采用矩阵方法进行的计算，其实可以看到形式会简洁很多很多

J = (1/2) * sum(sum((predict-Y).^2.*R)) + lambda / 2 * (sum(sum(Theta .^ 2)) + sum(sum(X .^ 2)));


%{
% 这一块是用非向量的方式写的，还是没有写出来这个有点复杂
for i = 1:size(X,1)
	for j = 1:size(X,2)
		X_grad(i,j) = sum((transpose(Theta(i,:)*transpose(X(i,:))) - Y(i,j)). * Theta(i,;));

	end

end

for j = 1:size(Theta,2)
	for i = 1:size(Theta,1)
		Theta_grad(i,j) = sum((Theta(i,:) * transpose(X(i,:)) - Y(i,j)).*X(i,;));

	end
end 
%}

% 这个是参考了别人的coding有点难度。
for i=1:num_movies
    idx = find(R(i, :) == 1);
    tempTheta = Theta(idx, :);
    tempY = Y(i, idx);
    X_grad(i, :) = (X(i, :) * tempTheta' - tempY) * tempTheta + lambda * X(i, :);
end

for i=1:num_users
    idx = find(R(:, i) == 1);
    tempX = X(idx, :);
    tempY = Y(idx, i);
    Theta_grad(i, :) = (tempX * Theta(i, :)' - tempY)' * tempX + lambda * Theta(i, :);
end







% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
