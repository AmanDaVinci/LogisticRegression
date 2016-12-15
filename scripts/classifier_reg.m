%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the X values and the third column
%  contains the label (y).

data = load('../data/ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);

plotData(X, y);

% Put some labels 
hold on;

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

% Specified in plot order
legend('y = 1', 'y = 0')
hold off;


%% =========== Regularized Logistic Regression ============

% Add Polynomial Features

% Note that mapFeature also adds a column of ones for us, so the intercept
% term is handled
X = mapFeature(X(:,1), X(:,2));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;

% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ============= Part 2: Regularization and Accuracies =============
%  Optional Exercise:
%  In this part, you will get to try different values of lambda and 
%  see how regularization affects the decision coundart
%
%  Try the following values of lambda (0, 1, 10, 100).
%
%  How does the decision boundary change when you vary lambda? How does
%  the training set accuracy vary?
%

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Experimenting with regularization parameter
lambdas = [0 0.5 1 1.5 2 5 10 100];

for i = 1:length(lambdas),
	% Set Options
	options = optimset('GradObj', 'on', 'MaxIter', 400);

	% Optimizing with lambda1
	[theta, J, exit_flag] = ...
				fminunc(@(t)(costFunctionReg(t, X, y, lambdas(i))), initial_theta, options);

	% Plot Boundary
	plotDecisionBoundary(theta, X, y);
	hold on;
	title(sprintf('lambda = %g', lambdas(i) ))

	% Labels and Legend
	xlabel('Microchip Test 1')
	ylabel('Microchip Test 2')

	legend('y = 1', 'y = 0', 'Decision boundary')
	hold off;

	% Compute accuracy on our training set
	p = predict(theta, X);

	fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
end

