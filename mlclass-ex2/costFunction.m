function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
result = 0;

A = 0;
B = 0;

grad = zeros(size(theta));

Z = 0;
T = 0;
TT = 0;
PP = 0;
SS = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
% Cost
Z = sigmoid(X*theta);
T = log(Z);
TT = -y.*T;

PP = 1 - Z;
T = log(PP);
SS = (1-y).*T;

result = (TT-SS);
result = sum(result);
J = result/m;

% Gradient Descent
A = (Z - y);   
B = A'*X;         
grad = B/m;             

% =============================================================

end
