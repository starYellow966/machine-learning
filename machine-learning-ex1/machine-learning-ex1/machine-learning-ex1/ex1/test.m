data = load('ex1data1.txt');
y = data(:, 2);
m = length(y)
X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1);% initialize fitting parameters

J = sum((X * theta - y).^2) / (2*m)
