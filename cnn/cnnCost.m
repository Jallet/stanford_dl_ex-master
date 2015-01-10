function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
                                filterDim,numFilters,poolDim,pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImages
%                array
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)


if ~exist('pred','var')
    pred = false;
end;


imageDim = size(images,1); % height/width of image
numImages = size(images,3); % number of images

%% Reshape parameters and setup gradient matrices

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias
[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,...
                        poolDim,numClasses);

% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
Wc_grad = zeros(size(Wc));
Wd_grad = zeros(size(Wd));
bc_grad = zeros(size(bc));
bd_grad = zeros(size(bd));

%%======================================================================
%% STEP 1a: Forward Propagation
%  In this step you will forward propagate the input through the
%  convolutional and subsampling (mean pooling) layers.  You will then use
%  the responses from the convolution and pooling layer as the input to a
%  standard softmax layer.

%% Convolutional Layer
%  For each image and each filter, convolve the image with the filter, add
%  the bias and apply the sigmoid nonlinearity.  Then subsample the 
%  convolved activations with mean pooling.  Store the results of the
%  convolution in activations and the results of the pooling in
%  activationsPooled.  You will need to save the convolved activations for
%  backpropagation.
convDim = imageDim-filterDim+1; % dimension of convolved output
outputDim = (convDim)/poolDim; % dimension of subsampled output

% convDim x convDim x numFilters x numImages tensor for storing activations
activations = zeros(convDim,convDim,numFilters,numImages);

% outputDim x outputDim x numFilters x numImages tensor for storing
% subsampled activations
activationsPooled = zeros(outputDim,outputDim,numFilters,numImages);

%%% YOUR CODE HERE %%%
lambda = 3e-3;
convolvedImage = zeros(convDim, convDim, numFilters, numImages);
poolPattern = (1 / poolDim ^ 2) * ones(poolDim);
for i = 1 : numImages
	for j = 1 : numFilters
		filter = rot90(squeeze(Wc(:, :, j)), 2);
		image = images(:, :, i);
		temp = conv2(image, filter,'valid');
		temp = temp + bc(j);
		convolvedImage(:, :, j, i) = 1 ./ (1 + exp(-temp));
		sizeImage = size(convolvedImage, 1);
		pooledImage = conv2(convolvedImage(:, :, j, i), poolPattern, 'valid');
		activationsPooled(:, :, j, i) = pooledImage(1 : poolDim : sizeImage, 1 : poolDim : sizeImage);
	end
end
%activationsPooled = 1 ./ (1 + exp(-activationsPooled));

% Reshape activations into 2-d matrix, hiddenSize x numImages,
% for Softmax layer
activationsPooled = reshape(activationsPooled,[],numImages);
%% Softmax Layer
%  Forward propagate the pooled activations calculated above into a
%  standard softmax layer. For your convenience we have reshaped
%  activationsPooled into a hiddenSize x numImages matrix.  Store the
%  results in probs.

% numClasses x numImages for storing probability that each image belongs to
% each class.
probs = zeros(numClasses, numImages); 
temp = exp(Wd * activationsPooled);
b = repmat(bd, 1, size(temp, 2));
temp = temp + b;
sumtemp = sum(temp, 1);
sumtemp = repmat(sumtemp, size(temp, 1), 1);
probs = temp ./ sumtemp;
%%% YOUR CODE HERE %%%

%%======================================================================
%% STEP 1b: Calculate Cost
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective.  Store your
%  results in cost.

%%% YOUR CODE HERE %%%

cost = 0; % save objective into cost
I = sub2ind(size(probs), labels', 1 : size(probs, 2));
f = log(probs(I));
cost = -sum(f(:)) / numImages;
weightDecayCost = (lambda / 2) * (sum(Wd(:) .^ 2) + sum(Wc(:) .^ 2));
cost = cost + weightDecayCost;
% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end;

%%======================================================================
%% STEP 1c: Backpropagation
%  Backpropagate errors through the softmax and convolutional/subsampling
%  layers.  Store the errors for the next step to calculate the gradient.
%  Backpropagating the error w.r.t the softmax layer is as usual.  To
%  backpropagate through the pooling layer, you will need to upsample the
%  error with respect to the pooling layer for each filter and each image.  
%  Use the kron function and a matrix of ones to do this upsampling 
%  quickly.

%%% YOUR CODE HERE %%%
delta = cell(4, 1);
delta{4} = zeros(numClasses, numImages);
nl = 4;
w = (1 : size(probs, 1))';
w = repmat(w, 1, size(probs, 2));
y = repmat(labels', size(w, 1), 1);
w = (w == y);
delta{4} = probs - w;


%delta{4} = sum(delta{4}, 2);
%delta{3} = Wd' * delta{4} .* activationsPooled .* (1 - activationsPooled);
delta{3} = Wd' * delta{4};
upPattern = 1 / (poolDim ^ 2) * ones(poolDim);
temp = reshape(delta{3}, outputDim, outputDim, numFilters, numImages);
delta{2} = zeros(convDim, convDim, numFilters, numImages);
for i = 1 : numImages
    for j = 1 : numFilters
        delta{2}(:, :, j, i) = kron(temp(:, :, j, i), upPattern);
        delta{2}(:, :, j, i) = delta{2}(:, :, j, i) .* convolvedImage(:, :, j, i) .* (1 - convolvedImage(:, :, j, i));
    end
end

%upPattern = ones(poolDim * poolDim, 1);
%delta{2} = kron(delta{3}, upPattern);
%delta{2} = delta{2} .* convolvedImage .* (1 - convolvedImage);
%%======================================================================
%% STEP 1d: Gradient Calculation
%  After backpropagating the errors above, we can use them to calculate the
%  gradient with respect to all the parameters.  The gradient w.r.t the
%  softmax layer is calculated as usual.  To calculate the gradient w.r.t.
%  a filter in the convolutional layer, convolve the backpropagated error
%  for that filter with each image and aggregate over images.

%%% YOUR CODE HERE %%%
Wd_grad = delta{4} * activationsPooled' ./ numImages + lambda * Wd;
bd_grad = sum(delta{4}, 2) ./ numImages;
for i = 1 : numImages
    for j = 1 : numFilters
       Wc_grad(:, :, j) = Wc_grad(:, :, j) + conv2(images(:, :, i), rot90(delta{2}(:, :, j, i), 2), 'valid');
       temp = delta{2}(:, :, j, i);
       bc_grad(j) = bc_grad(j) + sum(temp(:));
    end
end
bc_grad = bc_grad ./ numImages;
Wc_grad = Wc_grad ./ numImages + lambda * Wc;
%% Unroll gradient into grad vector for minFunc
grad = [Wc_grad(:) ; Wd_grad(:) ; bc_grad(:) ; bd_grad(:)];
end
