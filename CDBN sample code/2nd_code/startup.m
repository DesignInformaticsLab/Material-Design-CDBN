addpath utils/;
addpath function_code/;

% prepare CIFAR-10 dataset
prepare_2nd_layer;

savedir = 'results/';
if ~exist(savedir, 'dir'),
    mkdir(savedir);
end

rng('default');
