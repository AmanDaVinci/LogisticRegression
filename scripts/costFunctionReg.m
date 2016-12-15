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

% Number of features
n = size(theta);

% Computing the first summation in formula
for i = 1:m,
	z = X(i, :) * theta;
	h = sigmoid(z);
	J = J - y(i) * log(h) - (1 - y(i)) * log(1 - h);
end

% Simple Cost function
J = J / m; 

regParam = 0;
% Computing the regularization parameter
for i = 2:n,
	regParam = regParam + theta(i) * theta(i);
end
% Regularization Parameter
regParam = lambda * regParam / ( 2 * m);

% Regularized Cost Function
J = J + regParam;

% Computing the gradient
for j = 1:n,
	for i = 1:m,
		z = X(i, :) * theta;
		h = sigmoid(z);
		grad(j) = grad(j) + (h - y(i)) * X(i, j);
	end

	grad(j) = grad(j) / m;
	% Regularization for theta greater than 1
	if (j >= 2),
		grad(j) = grad(j) + lambda * theta(j) / m;
end
% =============================================================

end
