%% crbm_inference
function [poshidexp2] = crbm_inference(image, weight, Tlist, params,ii)
    rng(0);
    numchannels = 1; % weight.vishid assumes a single channel
    W = gather(weight.vishid);
    ws = sqrt(size(W,1));
    numhid = size(W,2); % number of filters
    numtx = size(Tlist,1);
    batchsize = (size(image,1)-ws+1)*(size(image,2)-ws+1);
%     batchsize = 1;
    numvis = params.numvis;
   

    hidprob = zeros(numhid, numtx+1, batchsize, 'single');
    % reshape for speedup
    vishid_up = W;
    vishid_down = W;
    if strcmp(params.intype, 'real'),
        vishid_up = 1/weight.sigma*vishid_up;
        vishid_down = weight.sigma*vishid_down;
    end
    hbiasmat = reshape(repmat(weight.hidbias(:,1:numtx), [1, 1, batchsize]), numhid, numtx*batchsize);
    vbiasmat = repmat(weight.visbias, [1, batchsize]);
    
    % get batches from a single image
%     xb = 
    
%     im2 = trim_image_for_spacing_fixconv(imdata, ws, spacing);
    [H,L] = size(image);
    
%     xb = reshape(image(10:ws+9,10:ws+9),[1,ws*ws]);
%     xb = patch.patch(1,:);
    num_patch = (L-ws+1)*(H-ws+1);
    xb = zeros(num_patch, ws^2);
    for i = 1 : (L-ws+1)
        for j = 1 : (H-ws+1)       
            temp = image(j:ws+j-1,i:ws+i-1);
            xb(j+(L-ws+1)*(i-1),:) = reshape(temp,[1,ws*ws]);       
        end    
    end
    
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
    Tlist_matrix = A; clear A;

    % positive phase       
    Tx = reshape(Tlist_matrix*xb', numvis, numtx*batchsize);
    hidprob = reshape(hidprob, numhid, numtx+1, batchsize);

    hidprob(:, 1:numtx, :) = gather(reshape(vishid_up'*Tx + hbiasmat, numhid, numtx, batchsize));
    hidprob(:, numtx+1, :) = 0;
    hidprob = exp(bsxfun(@minus, hidprob, max(hidprob, [], 2)));
    hidprob = bsxfun(@rdivide, hidprob, sum(hidprob, 2));
    
    % negative phase
    % hidden sampling
    hidprob = reshape(permute(hidprob, [2 1 3]), numtx+1, numhid*batchsize);
    [hidstate, ~] = sample_multinomial(hidprob, params.optgpu);
    hidstate = reshape(hidstate(1:numtx, :), numtx, numhid, batchsize);
    
    fname = sprintf('hidstates1st_WB_100_(2f12r6wsP10Pb01)_%d',ii);
    save(sprintf('1st_hidstate/%s.mat',fname),'hidstate', '-v7.3');
    
    negdata = zeros(L, H, numchannels);
    for nf = 1:numhid
        for nt = 1:numtx
            filter_t = gather(full(Tlist{nt})'*vishid_down(:,nf));
            filter_t = reshape(filter_t,[ws,ws,numchannels]);
            S = reshape(hidstate(nt,nf,:),L-ws+1,H-ws+1);
            temp = conv2_mult(S, filter_t, 'full');
            negdata = negdata + temp;
        end
%         if strcmp(params.intype, 'binary'),
%             negdata = sigmoid(negdata);
%         end
    end

        
    poshidexp2 = negdata;
    fprintf('Loading negdata %d...\n',ii);
%     figure(1);
%     display_network(reshape(negdata,size(negdata,1)*size(negdata,2),1));
    return
end
