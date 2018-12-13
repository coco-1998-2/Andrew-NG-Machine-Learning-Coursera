function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

mu = mean(X);
X_norm = bsxfun(@minus, X, mu);  
% not have bsxfun in Matlab 7.0, use repmat function instead.
%m = repmat(mu, size(X,1), 1);
%X_norm = minus(X, m);

sigma = std(X_norm);
X_norm = bsxfun(@rdivide, X_norm, sigma);
%m = repmat(sigma, size(X,1), 1);
%X_norm = rdivide(X, m);

% ============================================================

end
