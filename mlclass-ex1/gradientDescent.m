function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


A = 0;
AA = 0;

B = 0;
BB = 0;

temp1 = 0;
temp2 = 0;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

A = ((X*theta) - y);   % Gives me 5x1 vector
AA = sum(A);           % Adds the elements

B = X(:,2);           % Gives me 5x1 vector
BB = A'*B             % Automatically adds the elements

temp1 = theta(1)-((alpha*AA)/m);
temp2 = theta(2)-((alpha*BB)/m);

theta(1) = temp1;
theta(2) = temp2;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
