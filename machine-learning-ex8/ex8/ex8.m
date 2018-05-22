%% Machine Learning Online Class
%  Exercise 8 | Anomaly Detection and Collaborative Filtering
%     estimateGaussian.m
%     selectThreshold.m
%     cofiCostFunc.m
%% Initialization
clear ; close all; clc
%% ================== Part 1: Load Example Dataset  ===================
load('ex8data1.mat');
plot(X(:, 1), X(:, 2), 'bx');
axis([0 30 0 30]);
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');
%% ================== Part 2: Estimate the dataset statistics ===================
[mu sigma2] = estimateGaussian(X);
p = multivariateGaussian(X, mu, sigma2);
%  Visualize the fit
visualizeFit(X,  mu, sigma2);
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');
%% ================== Part 3: Find Outliers ===================
pval = multivariateGaussian(Xval, mu, sigma2);
[epsilon F1] = selectThreshold(yval, pval);
fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
fprintf('   (you should see a value epsilon of about 8.99e-05)\n\n');

outliers = find(p < epsilon);

%  Draw a red circle around those outliers
hold on
plot(X(outliers, 1), X(outliers, 2), 'ro', 'LineWidth', 2, 'MarkerSize', 10);
hold off
%% ================== Part 4: Multidimensional Outliers ===================
load('ex8data2.mat');
[mu sigma2] = estimateGaussian(X);
p = multivariateGaussian(X, mu, sigma2);
pval = multivariateGaussian(Xval, mu, sigma2);
[epsilon F1] = selectThreshold(yval, pval);
fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
fprintf('# Outliers found: %d\n', sum(p < epsilon));
fprintf('   (you should see a value epsilon of about 1.38e-18)\n\n');




