function XC = tirbm_inference(X, rs, weight, params, Tlist, M, P, optgpu, optsplit)

CIFAR_DIM = [32 32 3];

if ~exist('optsplit','var') || isempty(optsplit),
    optsplit = 'none';
end
if ~isempty(M) && ~isempty(P),
    whitening = 1;
else
    whitening = 0;
end

% compute features for all training images
if strcmp(optsplit,'split'),
    XC = zeros(size(X, 1), size(weight.vishid, 2)*4*2);
else
    XC = zeros(size(X, 1), size(weight.vishid, 2)*4);
end

if optgpu,
    weight.vishid = gpuArray(weight.vishid);
    weight.hidbias = gpuArray(weight.hidbias);
end

% weights
vishid = 1/weight.sigma*weight.vishid;
% hbiasmat = repmat(weight.hidbias, []);


% convert Tlist (cell) into matrix
A = zeros(size(Tlist{1}, 1)*params.numtx, size(Tlist{1}, 2), 'single');
if params.optgpu,
    A = gpuArray(A);
end

for i = 1:params.numtx,
    if params.optgpu,
        A((i-1)*size(Tlist{1}, 1)+1:i*size(Tlist{1}, 1), :) = gpuArray(single(full(Tlist{i})));
    else
        A((i-1)*size(Tlist{1}, 1)+1:i*size(Tlist{1}, 1), :) = full(Tlist{i});
    end
end
Tlist = A; clear A;

numvis = params.numvis;
numhid = params.numhid;
numtx = params.numtx;


hidprob = [];
hbiasmat = [];

tS = tic;
for i = 1:size(X, 1),
    if (mod(i, 1000) == 0),
        tE = toc(tS);
        fprintf('Extracting features: %d / %d, time:%g (min)\n', i, size(X, 1),tE/60);
        tS = tic;
    end
    
    % extract overlapping sub-patches into rows of 'patches'
    patches = [ im2col(reshape(X(i, 1:1024),CIFAR_DIM(1:2)), [rs rs]) ;
        im2col(reshape(X(i, 1025:2048),CIFAR_DIM(1:2)), [rs rs]) ;
        im2col(reshape(X(i, 2049:end),CIFAR_DIM(1:2)), [rs rs]) ]';
    
    % whiten
    if whitening
        % normalize for contrast
        patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches, 2)), sqrt(var(patches,[],2)+10));
        patches = bsxfun(@minus, patches, M) * P;
    end
    
    % compute features
    patches = patches';
    if optgpu,
        patches = gpuArray(patches);
    end
    
    if isempty(hidprob),
        batchsize = size(patches, 2);
        hidprob = zeros(numhid, numtx+1, batchsize, 'single');
        hbiasmat = repmat(weight.hidbias, [1 batchsize]);
        if params.optgpu,
            hidprob = gpuArray(hidprob);
            hbiasmat = gpuArray(hbiasmat);
        end
    end
    
    hidprob_mult = tirbm_inference_sub(Tlist, patches, vishid, hbiasmat, hidprob, numvis, numtx, numhid);
    if strcmp(optsplit,'split'),
        nhidprob_mult = tirbm_inference_sub(Tlist, -patches, vishid, hbiasmat, hidprob, numvis, numtx, numhid);
        feat = [hidprob_mult ; nhidprob_mult]';
    else
        feat = hidprob_mult';
    end
    
    % reshape to K-channel image
    prows = CIFAR_DIM(1)-rs+1;
    pcols = CIFAR_DIM(2)-rs+1;
    feat = reshape(feat, prows, pcols, numel(feat)/prows/pcols);
    
    % pool over quadrants
    halfr = round(prows/2);
    halfc = round(pcols/2);
    q1 = sum(sum(feat(1:halfr, 1:halfc, :), 1),2);
    q2 = sum(sum(feat(halfr+1:end, 1:halfc, :), 1),2);
    q3 = sum(sum(feat(1:halfr, halfc+1:end, :), 1),2);
    q4 = sum(sum(feat(halfr+1:end, halfc+1:end, :), 1),2);
    
    % concatenate into feature vector
    XC(i,:) = [q1(:); q2(:); q3(:); q4(:)]';
end


function [hidprob_mult, hidprob] = tirbm_inference_sub(Tlist, xb, vishid, hbiasmat, hidprob, numvis, numtx, numhid)

batchsize = size(xb, 2);

Tx = reshape(Tlist*xb, numvis, numtx*batchsize);
hidprob = reshape(hidprob, numhid, numtx+1, batchsize);
hidprob(:, 1:numtx, :) = reshape(vishid'*Tx + hbiasmat, numhid, numtx, batchsize);
hidprob(:, numtx+1, :) = 0;
hidprob = exp(bsxfun(@minus, hidprob, max(hidprob, [], 2)));
hidprob = bsxfun(@rdivide, hidprob, sum(hidprob, 2));
hidprob_mult = double(squeeze(sum(gather(hidprob(:,1:numtx,:)),2)));

return;

