% -----------------------------------------------------------------------
%   load small patch of (ws x ws) from CIFAR-10 dataset
% -----------------------------------------------------------------------

% function [xtr, patch, M, P] = load_patches(npatch, ws)
function [xtr, patch] = load_patches(npatch, ws)

prepare_cifar10;

% Load CIFAR training data
fprintf('Loading training data...\n');
f1=load([CIFAR_DIR '/WB.mat']);

xtr = double([f1.WB;]);
xtr = xtr(1:10,:);
clear f1;
fname = sprintf('WB_f2rot12_ws%d',ws);

if ~exist('patch','dir'),
    mkdir('patch');
end

try
    load(sprintf('patch/%s.mat', fname),'patch');
catch
    if npatch > 0,

        patch = zeros(npatch, ws*ws);
        for i=1:npatch
            if (mod(i,1000) == 0),
                fprintf('Extracting patch: %d / %d\n', i, npatch);
            end
            

            r = random('unid', CIFAR_DIM(1) - ws + 1);
            c = random('unid', CIFAR_DIM(2) - ws + 1);

            cpatch = reshape(xtr(mod(i-1,size(xtr,1))+1, :), CIFAR_DIM);
            cpatch = cpatch(r:r+ws-1,c:c+ws-1);
%             patch(i,:) = cpatch(:)';
            k=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             while sum(sum(cpatch))>50 && sum(cpatch(:,1))==12 && sum(cpatch(1,:))==12%&& sum(sum(cpatch))<120 % less than 1%               
%                 r = random('unid', CIFAR_DIM(1) - ws + 1);
%                 c = random('unid', CIFAR_DIM(2) - ws + 1);
%                 cpatch = reshape(xtr(mod(i-1,size(xtr,1))+1, :), CIFAR_DIM);
%                 cpatch = cpatch(r:r+ws-1,c:c+ws-1);
%                 k=k+1;
%                 if k > 200
%                     break
%                 end
% 
%             end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
            patch(i,:) = cpatch(:)';
        end
        
        % normalize for contrast
        patch = bsxfun(@rdivide, bsxfun(@minus, patch, mean(patch,2)), sqrt(var(patch,[],2)+10));
        
        % whiten
%         C = cov(patch);
%         M = mean(patch);
%         [V,D] = eig(C);
%         P = V * diag(sqrt(1./(diag(D) + 0.1))) * V';
%         patch = bsxfun(@minus, patch, M) * P;
        
    else
        patch = [];
        M = [];
        P = [];
    end
    patch = single(patch);
    save(sprintf('patch/%s.mat', fname), 'patch','-v7.3');
end

patch = double(patch);

return