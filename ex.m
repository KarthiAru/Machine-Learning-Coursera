%% Exercise 1A: Single Linear Regression

% Initialization
clear ; close all; clc

data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples

fprintf('Plotting Data ...\n')
plot(X, y, 'rx', 'MarkerSize', 10);
ylabel('Profit in $10,000s');
xlabel('Population of City in 10,000s');

fprintf('Running Gradient Descent ...\n')
X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(size(X, 2), 1); % initialize fitting parameters
% Some gradient descent settings
num_iters = 5000;
alpha = 0.01;
J = zeros(num_iters, 1);
% run gradient descent
for i = 1: num_iters
    theta = theta - (alpha*(1/m)*(X'*((X*theta)-y)));
    sse = sum(((X*theta)-y).^2);
    J(i) = sse/(2*m);
end

iterations=1:num_iters;
plot(iterations,J,'-')

% print theta to screen
fprintf('Theta found by gradient descent: ');
fprintf('%f %f \n', theta(1), theta(2));

% Plot the linear fit
plot(X(:,2), y, 'rx', 'MarkerSize', 10); % Plot the data
ylabel('Profit in $10,000s'); % Set the y-axis label
xlabel('Population of City in 10,000s'); % Set the x-axis label
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] *theta;
fprintf('For population = 35,000, we predict a profit of %f\n',predict1*10000);
predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n',predict2*10000);

%% Exercise 1B: Multiple Linear Regression (Gradient Descent)

% Initialization
clear ; close all; clc

fprintf('Solving with Gradient Descent ...\n');

% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
mu = mean(X);
sigma = std(X);
for i = 1:size(X,2)
X(:,i) = (X(:,i) - mu(:,i))/sigma(:,i);
end

fprintf('Running Gradient Descent ...\n')
X = [ones(m, 1), X]; % Add a column of ones to x
theta = zeros(size(X, 2), 1); % initialize fitting parameters
% Some gradient descent settings
num_iters = 400;
alpha = 0.01;
J = zeros(num_iters, 1);
% run gradient descent
for i = 1: num_iters
    theta = theta - (alpha*(1/m)*(X'*((X*theta)-y)));
    sse = sum(((X*theta)-y).^2);
    J(i) = sse/(2*m);
end

iterations=1:num_iters;
plot(iterations,J,'-')

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Predict values for 1650 sq-ft and 3 BHK
price = theta(1) + theta(2)*((1650-mu(1))/sigma(1)) + theta(3)*((3-mu(2))/sigma(2));
fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

%% Exercise 1C: Multiple Linear Regression (Normal Equations)

% Initialization
clear ; close all; clc

fprintf('Solving with normal equations...\n');

data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = zeros(size(X, 2), 1);
theta = pinv(X'*X)*X'*y;

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Prediction
price = theta(1) + theta(2)*1650 + theta(3)*3;

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);
     
%% Exercise 2A: Logistic Regression

% Initialization
clear ; close all; clc

% Load Data
data = load('ex2data1.txt');
X = data(:, [1, 2]); y = data(:, 3);


fprintf(['Plotting data with + indicating (y = 1) examples and o ' ...
         'indicating (y = 0) examples.\n']);

% Plot
pos = find(y==1); neg = find(y == 0);

plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7)
hold on;
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7)
xlabel('Exam 1 score')
ylabel('Exam 2 score')

fprintf('Running Gradient Descent ...\n')

[m, n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];

% Initialize fitting parameters
theta = zeros(n + 1, 1);

% Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), theta, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

% Plot Boundary
plotDecisionBoundary(theta, X, y);
hold on;
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;

%Predict
prob = 1 ./ (1+exp(-1*([1 45 85] * theta)));
fprintf(['For a student with scores 45 and 85, we predict an admission ' ...
         'probability of %f\n\n'], prob);

% Compute accuracy on our training set
h = 1 ./ (1+exp(-1*(X * theta)));
p = h >= 0.5;

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

%% Exercise 2A: Logistic Regression with Regularization

% Initialization
clear ; close all; clc

% Load Data
data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);

% Plot
pos = find(y==1); neg = find(y == 0);

plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7)
hold on;
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7)
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0')

% Map features
X1 = X(:,1); X2 = X(:,2);
degree = 6;
X = ones(size(X1(:,1)));
for i = 1:degree
    for j = 0:i
        X(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;

% Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

% Plot Boundary
plotDecisionBoundary(theta, X, y);
title(sprintf('lambda = %g', lambda))
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

% Compute accuracy on our training set
h = 1 ./ (1+exp(-1*(X * theta)));
p = h >= 0.5;
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

%% Exercise 3 | Part 1: One-vs-all


%% Exercise 3 | Part 2: Neural Networks