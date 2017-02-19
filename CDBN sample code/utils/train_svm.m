% -----------------------------------------------------------------------
%   Train SVM with hinge loss using minFunc
% 
%   xtr     : M x N, M: number of examples, N: input dimension
%   ytr     : M x 1
% -----------------------------------------------------------------------

function theta = train_svm(xtr, ytr, C)

addpath(genpath('utils/minFunc_2012/'));
warning off all;

% reorder the data
if size(xtr, 1) ~= length(ytr), 
    xtr = xtr';
end
ytr = ytr(:);

nc = max(ytr);
w = zeros(size(xtr, 2)*nc, 1);

% update with full batch
w = minFunc(@my_l2svmloss, w, struct('MaxIter', 1000, 'MaxFunEvals', 1000, 'Display', 'off'), xtr, ytr, nc, C);
theta = reshape(double(w), size(xtr, 2), nc);

return;

% 1-vs-all L2-svm loss function;  similar to libLinear.
function [loss, g] = my_l2svmloss(w, X, y, K, C)
[M, N] = size(X);
theta = reshape(w, N, K);
Y = bsxfun(@(y, ypos) 2*(y==ypos)-1, y, 1:K);

margin = max(0, 1 - Y .* (X*theta));
loss = (0.5 * sum(theta.^2)) + C*mean(margin.^2);
loss = sum(loss);
g = theta - 2*C/M * (X' * (margin .* Y));
g = g(:);

return
