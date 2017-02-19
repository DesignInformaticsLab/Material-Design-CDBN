% -------------------------------------------------------------------------
% save weights, accumulated gradient, params, history
% -------------------------------------------------------------------------


function [weights, grad] = save_params(fname_mat, weights, grad, params, t, history)

weights = gpu2cpu_struct(weights);
grad = gpu2cpu_struct(grad);
history = gpu2cpu_struct(history);

save(fname_mat, 'weights', 'grad', 'params', 't', 'history');

return;