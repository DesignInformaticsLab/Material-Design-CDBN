addpath utils/;

% prepare CIFAR-10 dataset
prepare_cifar10;

savedir = 'results/';
if ~exist(savedir, 'dir'),
    mkdir(savedir);
end

rng('default');
