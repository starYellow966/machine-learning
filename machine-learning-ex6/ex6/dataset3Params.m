function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_list = [0.01,0.03,0.1,0.3,1,3,10,30]; % 
sigma_list = [0.01,0.03,0.1,0.3,1,3,10,30];
min = [-1,0,0];
for i = 1:size(C_list(:),1)
  for j = 1:size(sigma_list(:),1)
    model= svmTrain(X, y, C_list(i), @(x1, x2) gaussianKernel(x1, x2, sigma_list(j)));
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));
    if min(1) == -1 || min(1) > error
      min = [error, C_list(i), sigma_list(j)];
    end
  endfor
endfor

C = min(2)
sigma = min(3)
% =========================================================================

end
