function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
result = 0;

thetta = 0;


grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


result = ((X*theta) - y).^2;
result = sum(result);
result = (0.5*result)/m;

thetta = theta;               % to store theta in thetta since theta's values are changing in the remaining statements
theta = theta(2:end);
theta =  theta.*theta;
theta =  sum(theta);
theta = (0.5*lambda*theta) / m;

J = result + theta;



result = (X*thetta) - y;
result = result'*X;
result = result/m;

thetta = thetta(2:end);
thetta = (thetta*lambda) / m;
thetta = [0 ; thetta]';

grad = result + thetta;


% =========================================================================

grad = grad(:);

end
