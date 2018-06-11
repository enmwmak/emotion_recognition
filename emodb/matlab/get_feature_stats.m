function [mu, sigma] = get_feature_stats(x)

% Find feature stats
mu = mean(x,1);
sigma = std(x,1,1);

