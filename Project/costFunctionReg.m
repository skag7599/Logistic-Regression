function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples


J = 0;
grad = zeros(size(theta));



W=X*theta;
W=sigmoid(W);
s=0;
for i=1: m
l1=log(W(i));
l2=log(1-W(i));
s=s+-y(i)*l1-(1-y(i))*(l2);
end;
s=s/m;
re=0;
for i=2: size(theta)
re=re+theta(i)*theta(i);
end;
re=re/m;
re=re/2;
re=re*lambda;
J=s+re;


i=1;
grad(i)=0;

 for j=1:m
 grad(i)=grad(i)+(W(j)-y(j))*X(j,i);
 end;
grad(i)=grad(i)/m;



for i=2: size(theta)
grad(i)=0;

 for j=1:m
 grad(i)=grad(i)+(W(j)-y(j))*X(j,i);
 end;
 grad(i)=grad(i)+lambda*theta(i);
grad(i)=grad(i)/m;
end;




% =============================================================

end
