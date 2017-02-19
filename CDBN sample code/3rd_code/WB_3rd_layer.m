function [acc, weight, params] = WB_3rd_layer...
    (optgpu, rs, numhid, pbias, plambda, eta_sigma, split)
%     (optgpu, rs, numhid, txtype, numtx, numrot, grid, pbias, plambda, eta_sigma, split)
       

startup;

%  dataset = 'cifar10';
dataset = 'alloy';
intype = 'binary';
sptype = 'exact';
epsilon = 0.002; %original 0.005, change 0.001 to 0.0001
batchSize = 200; 
if ~exist('split', 'var'),
    split = 'split'; end

% hyper parameters
if ~exist('optgpu', 'var'),
    optgpu = 0; end
if ~exist('rs', 'var'),
    rs = 6; end
if ~exist('numhid', 'var'),
    numhid = 200; end
if ~exist('grid', 'var'),
    grid = 1; end
if ~exist('numtx', 'var'),
    numtx = 1; end
if ~exist('numrot', 'var'),
    numrot = 1; end
if ~exist('txtype', 'var'),
    txtype = 'trans'; end
if ~exist('pbias', 'var'),
    pbias = 0.1; end
if ~exist('plambda', 'var'),
    plambda = 3; end
if ~exist('eta_sigma', 'var'),
    eta_sigma = 0.01; end


% -----------------------------------------------------------------------
%                                       hyper parameters for dataset
% -----------------------------------------------------------------------

params.dataset = dataset;
params.rs = rs;
params.numhid = numhid;
params.optgpu = optgpu;
params.grid = grid;
params.txtype = txtype;
params.numtx = numtx;
params.numrot = numrot;
params.dataset = dataset;
params.pbias = pbias;
params.plambda = plambda;
params.intype = intype;
params.sptype = sptype;
params.savepath = savedir;

% other hyper parameters
params.maxiter = 2000;
params.batchsize = batchSize;
params.epsilon = epsilon;
params.eta_sigma = eta_sigma;
params.l2reg = 1e-4;
params.epsdecay = 0.01;
params.kcd = 1;
params.numch = 40;


% -----------------------------------------------------------------------
%                                   load training and testing images
% -----------------------------------------------------------------------

npatch = 5000; %%% test,original 12000
[xtrain, patches] = load_patches_3rd_layer(npatch, params.rs);
% [patches]=noload_patches(params.rs);


% -----------------------------------------------------------------------
%                                   generate transformation matrices
% -----------------------------------------------------------------------

params.ws = params.rs - (params.numtx-1)*params.grid;
params.scales = params.ws:params.grid:params.rs;
params.rSize = params.rs^2*params.numch;
params.numvis = params.ws^2*params.numch;

% Tlist = get_txmat(params.txtype, params.rs, params.ws, params.grid, params.numrot, params.numch);
% params.numtx = length(Tlist);


% -----------------------------------------------------------------------
%                                                         train TIRBM
% -----------------------------------------------------------------------

% filename to save
if strcmp(params.txtype, 'rot'),
    fname = sprintf('WB_45th_(2f40f144f288f6w9ws9ws12ws)_%s_w%d_b%02d_%s_nrot%d_pb%g_pl%g', ...
        params.dataset, params.ws, params.numhid, params.txtype, params.numtx, params.pbias, params.plambda);
% trans and scale will be ignored here
elseif strcmp(params.txtype, 'trans') || strcmp(params.txtype, 'scale'),
    fname = sprintf('WB_45th_(2f40f144f288f6w9ws9ws12ws)_%s_w%d_b%02d_%s_ntx%d_gr%d_pb%g_pl%g', ...
        params.dataset, params.ws, params.numhid, params.txtype, params.numtx, params.grid, params.pbias, params.plambda);
end
params.fname  = sprintf('%s/%s', params.savepath, fname);
% params.fname = sprintf('2nd_layer_training');

try
    load([params.fname '_iter_' num2str(params.maxiter) '.mat'], 'weight', 'params');
catch
%     [weight, params] = training_2nd_layer(patches', params, Tlist);
    [weight, params] = training_3rd_layer(patches', params);
end
clear patches;

acc = [];
fprintf('the dir/name of the saved file is-----> \n%s_iter_%d.mat\n',params.fname,params.maxiter);
return;
