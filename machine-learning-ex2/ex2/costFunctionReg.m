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
%tempG=grad;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%temp1=-y.*(log(sigmoid(X*theta)))-(ones(m,1)-y).*(log(ones(m,1)-sigmoid(X*theta)));
%temp2=(lambda/m/2)*((ones(1,size(theta)))*(theta.^2));
%for i=1:size(theta)
%  if i==1
%      tempG(i)=(X(:,i)'*(sigmoid(X*theta)-y))/m;
%  else
%      tempG(i)=(X(:,i)'*(sigmoid(X*theta)-y))/m+lambda/m*theta(i);
%end
%tempG=(X'*(sigmoid(X*theta)-y))/m;
%temp3=lambda/m*(theta);
%temp2(1)=0;
%temp3(1)= ;
%grad=tempG+temp3;
%J=((ones(m,1)'*temp1)/m)+temp2;
h = sigmoid(X*theta);
J = mean(-y.*log(h)-(1-y).*log(1-h)) ...
    + lambda/(2*m)*sum(theta(2:end).^2);
grad = ((X'* (h - y) + lambda*[0;theta(2:end)])) / m;






% =============================================================

end
