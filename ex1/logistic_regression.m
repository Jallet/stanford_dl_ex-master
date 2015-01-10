function [f,g] = logistic_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %

  m=size(X,2);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));


  %
  % TODO:  Compute the objective function by looping over the dataset and summing
  %        up the objective values for each example.  Store the result in 'f'.
  %
  % TODO:  Compute the gradient of the objective by looping over the dataset and summing
  %        up the gradients (df/dtheta) for each example. Store the result in 'g'.
  %
%%% YOUR CODE HERE %%%

for i = (1 : size(X, 2))
    hthetax = 1 / (1 + exp(-1 * theta' * X(1 : size(X, 1), i)));
    f = f + y(i) * log(hthetax) + (1 - y(i)) * log(1 - hthetax);
end
f = -1 * f;

for i = (1 : size(X, 2))
    for j = (1 : size(g, 1))
        g(j) = g(j) + X(j, i) * (hthetax - y(i));
    end
end