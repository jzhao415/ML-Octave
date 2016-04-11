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
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
% Add a column 1s to X
X = [ones(size(X, 1), 1) X];
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

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

% Part 1
for i=1:m
  yk=zeros(num_labels,1);
  yk(y(i)) = 1;
  ak2 = zeros(hidden_layer_size,1);
  for kai=1:hidden_layer_size
        ak2(kai) = Theta1(kai,:)*X(i,:)';
        ak2(kai) = sigmoid(ak2(kai));
  end
  ak2 = [1;ak2];
  ak3 = zeros(num_labels,1);
  for kai=1:num_labels
        ak3(kai) = Theta2(kai,:)*ak2;
        ak3(kai) = sigmoid(ak3(kai));
  end
  for k=1:num_labels
        J = J + (-yk(k))*log(ak3(k)) - (1 - yk(k))*log(1-ak3(k));
  end
end

J = J/m;

T1 = 0;
for i=1:size(Theta1, 1)
    for k=2:size(Theta1,2)
       T1 = T1 + Theta1(i,k).^2;
    end
end
T2 = 0;
for i=1:size(Theta2,1)
     for k=2:size(Theta2,2)
       T2 = T2 + Theta2(i,k).^2;
     end
end

J = J + lambda*(T1+T2)/(2*m);

% Part 2
delta_1 = 0;
delta_2 = 0;
for t = 1:m
  yk=zeros(num_labels,1);
  yk(y(t)) = 1;
  %step1
  a_1 = zeros(size(X,2),1)';
  a_1 = X(t,:)'; %a_1 size 401,1
  a_2 = zeros(hidden_layer_size,1);
  
  for kai=1:hidden_layer_size
        z_2(kai) = Theta1(kai,:)*X(t,:)';
        a_2(kai) = sigmoid(z_2(kai));
  end
  a_2 = [1 ; a_2];
  a_3 = zeros(num_labels,1);
  for kai=1:num_labels
        z_3(kai) = Theta2(kai,:)*a_2;
        a_3(kai) = sigmoid(z_3(kai));
  end
  %step2 
  sigma_3 = a_3 - yk; %size 10,1
  %step3
  sigma_2 = (sigma_3'*Theta2.*sigmoidGradient([ones(size(z_2,1),1) z_2]))(2:end); %size 25,1
  %step4
  delta_1 = delta_1 + sigma_2'*a_1';
  delta_2 = delta_2 + sigma_3*a_2';

end
  %step 5
  Theta1_grad = delta_1./m + lambda/m*[zeros(size(Theta1,1),1) Theta1(:,2:end)];
  Theta2_grad = delta_2./m + lambda/m*[zeros(size(Theta2,1),1) Theta2(:,2:end)];


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
