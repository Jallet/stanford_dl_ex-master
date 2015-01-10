function [f,g] = linear_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The target value for each example.  y(j) is the target for example j.
  %
  m=size(X,2);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));
  temp = theta' * X;
  temp = temp - y;
  f = 0.5 * temp .^ 2;
  f = sum(f(:));
  %
  % TODO:  Compute the linear regression objective function and gradient 
  %        using vectorized code.  (It will be just a few lines of code!)
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %
%%% YOUR CODE HERE %%%

g = X * ((theta' * X) - y)';

temptheta = repmat(theta, 1, size(theta, 1));
tempy = repmat(y, size(theta, 1), 1);
epsilon = 10 ^ -4;
threshold = 10 ^ -3;
epsilonm = epsilon * eye(size(theta, 1));
theta1 = temptheta + epsilonm;
theta2 = temptheta - epsilonm;
f1 = (theta1' * X - tempy) .^ 2;
f1 = 0.5 * sum(f1, 2);
f2 = (theta2' * X - tempy) .^ 2;
f2 = 0.5 * sum(f2, 2);
tempg = (f1 - f2) ./ (2 * epsilon);
g;
tempg = abs((tempg - g) ./ g);
if any(tempg > threshold)
    fprintf('\n\n\ngradient is not correct.............................................................\n\n\n')
end