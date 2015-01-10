function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
  temp = theta' * X;
  temp = exp(temp);
  sumtemp = sum(temp, 1) + 1;
  %fprintf('size of sumtemp is\n')
  sumtemp = repmat(sumtemp, num_classes, 1);
  last_row = ones(1, m);
  temp = [temp; last_row];
  p = temp ./ sumtemp;
  I = sub2ind(size(p), y, 1 : size(p, 2));
  %values = p(I);
  f = log(p(I));
  f = -1 * sum(f(:));
  
  w = (1 : (num_classes - 1))';
  w = repmat(w, 1, size(X, 2));
  temp = repmat(y, size(w, 1));
  w = (temp == w);
  g = X * w';
  %for k = (1 : (num_classes - 1))
  %  w = (y == k);
  %  w = w - p(k, :);
  %  w = repmat(w, size(X, 1), 1);
  %  temp = w .* X;
  %  g( : , k) = -1 * sum(temp, 2);
  %  %g( : , k) = temp;
  %end
  g=g(:); % make gradient a vector for minFunc

