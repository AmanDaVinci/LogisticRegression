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

%% ============= Experimental Regularization and Accuracies =============

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Experimenting with regularization parameter with steps of 3
lambdas = [0 0.0001 0.0003 0.0010 0.0030 0.01 0.03 0.1 0.3 1 3 10 30 60 100 130 160 170 173 176 180 300 1000];

for i = 1:length(lambdas),
	% Set Options
	options = optimset('GradObj', 'on', 'MaxIter', 400);

	% Optimizing with lambda1
	[theta, J, exit_flag] = ...
				fminunc(@(t)(costFunctionReg(t, X, y, lambdas(i))), initial_theta, options);

	% Compute accuracy on our training set
	p = predict(theta, X);
	accuracy = mean(double(p == y)) * 100;
	fprintf('Train Accuracy: %f\n', accuracy);

	% Plot Boundary
	plotDecisionBoundary(theta, X, y);
	hold on;
	title(sprintf('Accuracy: %f for lambda = %g', accuracy, lambdas(i) ))

	% Labels and Legend
	xlabel('Microchip Test 1')
	ylabel('Microchip Test 2')

	legend('y = 1', 'y = 0', 'Decision boundary')
	hold off;

end

