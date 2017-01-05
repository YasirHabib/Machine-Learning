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

result = 0;

A = 0;
B = 0;

Z = 0;
T = 0;
TT = 0;
PP = 0;
SS = 0;

XX = 0;
Regularised = 0;

thetta = 0;
YY = 0;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% cost
Z = sigmoid(X*theta);
T = log(Z);
TT = -y.*T;

PP = 1 - Z;
T = log(PP);
SS = (1-y).*T;

result = (TT-SS);
result = sum(result);
XX = result/m;

thetta = theta;               % to store theta in thetta since theta's values are changing in the remaining statements
theta = theta(2:end);
theta =  theta.*theta;
theta =  sum(theta);
Regularised = (0.5*lambda*theta) / m;

J = XX + Regularised;


% Gradient Descent
A = (Z - y);   
B = A'*X;         
YY = B/m;

thetta = thetta(2:end);
thetta = (thetta*lambda) / m;

thetta = [0 ; thetta]';

grad = YY + thetta;
% =============================================================

end
