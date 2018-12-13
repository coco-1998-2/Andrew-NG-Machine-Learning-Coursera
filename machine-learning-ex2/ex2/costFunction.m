function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%
A = sigmoid((theta'*X')');
%之前写的比较傻，直接用向量化会简洁
%pos = find(y==1) 
%neg = find(y==0)
%part1 = sum(-y(pos).*(log(A)(pos)));
%part2 = sum(-(1-y(neg)).*log(1-A(neg)));
%J = (part1 + part2)/m;
J = sum(-y.*log(A) - (1-y).*(log(1-A)))/m;

grad = ((A-y)'*X)'/m;








% =============================================================

end
