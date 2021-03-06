function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
J=s;


for i=1: size(theta)
grad(i)=0;
 for j=1:m
 grad(i)=grad(i)+(W(j)-y(j))*X(j,i);
 end;
grad(i)=grad(i)/m;
end;







%
%htheta=zeros(size(theta));
%for i=1:m
%htheta(i)=X(i)*theta;
%end

%htheta=sigmoid(htheta);
%lhtheta=log(htheta);
%dlhtheta=1-lhtheta;
%dy=1-y;
%J=(-y'*lhtheta-dy'*dlhtheta);
%J=J/m;


%




% =============================================================

end
