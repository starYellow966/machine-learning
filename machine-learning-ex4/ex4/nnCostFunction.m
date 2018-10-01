function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1)); % 25*401

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1)); % 10*26

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1)); %25*401
Theta2_grad = zeros(size(Theta2)); %10*26

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% 按样本遍历
for i=1:m
  temp_y = zeros(size(Theta2,1), 1); % 将输出修改为向量
  temp_y(y(i)) = 1;

  % 3层神经网络向前传播,计算h(x).
  z2 = Theta1 * [1 X(i, :)]'; % 25*1
  a2 = sigmoid(z2);% 增加偏置单元
  z3 = Theta2 * [1;a2]; % 10*1
  a3 = sigmoid(z3);% 增加偏置单元
  
  J = J + sum(temp_y' * log(a3) + (1 .- temp_y)' * log(1 .- a3));
  
  % 反向传播算法
  error_term3 = zeros(size(temp_y)); % 10*1
  error_term3 = a3 .- temp_y;
  error_term2 = zeros(size(Theta1)); % 26*401
  error_term2 = (Theta2' * error_term3) .* [1;sigmoidGradient(z2)]; %todo
  error_term2 = error_term2(2:end); %todo
  Theta1_grad = Theta1_grad + error_term2 * [1 X(i,:)]; %todo
  Theta2_grad = Theta2_grad + error_term3 * [1 a2']; % todo
endfor

% 代价正则化
J = -1 / m * J; % 计算代价
temp_theta1 = Theta1;
temp_theta2 = Theta2;
temp_theta1(:,1) = 0;
temp_theta2(:,1) = 0;
J = J + lambda / (2*m) * (sum(sum(temp_theta1 .^2)) + sum(sum(temp_theta2.^2))); 

% 反向传播算法,正则化
Theta1_grad = Theta1_grad / m + lambda/m * Theta1;
Theta1_grad(:,1) = Theta1_grad(:,1) - lambda/m * Theta1(:,1);
Theta2_grad = Theta2_grad / m + lambda/m * Theta2;
Theta2_grad(:,1) = Theta2_grad(:,1) - lambda/m * Theta2(:,1);
grad = [ Theta1_grad(:);Theta2_grad(:)];














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
