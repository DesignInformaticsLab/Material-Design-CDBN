function [weights, grad] = update_params(weights, grad, pos, neg, momentum, lr)

fname = fieldnames(weights);

for i = 1:length(fname),
    % load fields
    pA = getfield(pos, fname{i});
    nA = getfield(neg, fname{i});
    gA = getfield(grad, fname{i});
    A = getfield(weights, fname{i});
    
    gA = momentum*gA + lr*(pA - nA);
    A = A + gA;
    
    % update accumulate parameter and weights
    grad = setfield(grad, fname{i}, gA);
    weights = setfield(weights, fname{i}, A);
end

return;
