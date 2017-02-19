% -----------------------------------------------------------------------
%   load small patch of (ws x ws) from CIFAR-10 dataset
% -----------------------------------------------------------------------

function [xtr, patch, M, P] = load_patches_3rd_layer(npatch, ws)

prepare_3rd_layer;

% Load CIFAR training data
fprintf('Loading training data...\n');
% for ii = 1:99
% fname = sprintf('hidstates_%d',ii);
% f1=load(sprintf(CIFAR_DIR '/%s.mat', fname));   
% temp = double([f1.hidstate;]);
% xtr(ii,:) = temp(:)';
% % f1=load([CIFAR_DIR '/filter8_ws12.mat']);
% end
f1=load([CIFAR_DIR '/hidstates2nd_10_WB_pooled.mat']);
xtr = double([f1.xtr;]);
% xtr = gpuArray(xtr);

% xtr = double([f1.poshidstates;]);
% xtr = double([f1.hidstate;]);
% ytr = double([f1.labels;]) + 1; % add 1 to labels!
% xtr = permute(xtr,[1,2,3]);
% with rotation (data get from max's method)
% xtr = reshape(xtr,[size(xtr,1),size(xtr,2)*size(xtr,3)]);
% no rotation (data get from ruijin's method)
% xtr = reshape(xtr,[size(xtr,1),size(xtr,2)*size(xtr,3)]);


clear f1;
% fname = sprintf('hidstate_3rd_10_2f40f288f_ws6ws9ws%d',ws);
fname = sprintf('hidstate_45th_2f40f144f288f_ws6ws9ws9ws%d',ws);

if ~exist('patch','dir'),
    mkdir('patch');
end

try
    load(sprintf('patch/%s.mat', fname));
catch
    if npatch > 0,
        % extract random patch
        patch = zeros(npatch, ws*ws*CIFAR_DIM(3)); 
        for i=1:npatch
            if (mod(i,1000) == 0),
                fprintf('Extracting patch: %d / %d\n', i, npatch);
            end
            r = random('unid', CIFAR_DIM(1) - ws + 1);
            c = random('unid', CIFAR_DIM(2) - ws + 1);
            cpatch = reshape(xtr(mod(i-1,size(xtr,1))+1, :), CIFAR_DIM);
            cpatch = cpatch(r:r+ws-1,c:c+ws-1,:); 
            patch(i,:) = cpatch(:)';
        end
        

%         
    else
        patch = [];
        M = [];
        P = [];
    end
    patch = single(patch);
    save(sprintf('patch/%s.mat', fname), 'patch', '-v7.3');
end

patch = double(patch);

return