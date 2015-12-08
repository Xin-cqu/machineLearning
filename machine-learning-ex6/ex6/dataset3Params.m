function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
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
c_vec=[0.01,0.03,0.1,0.3,1,3,10,30];
sigma_vec=[0.01,0.03,0.1,0.3,1,3,10,30];
i_index=0;
j_index=0;
pre_error=zeros(8,8);
for i=1:8
  for j=1:8
    C=c_vec(i);
    sigma=sigma_vec(j);
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    prediction=svmPredict(model,Xval);
    pre_error(i,j)=mean(double(prediction~=yval));
  end
end
min_error=min(min(pre_error));
[i_index,j_index]=find(pre_error==min_error);
C=c_vec(i_index);
sigma=sigma_vec(j_index);

% what needs to improve is to select c and sigma in a large collection set.





% =========================================================================

end
