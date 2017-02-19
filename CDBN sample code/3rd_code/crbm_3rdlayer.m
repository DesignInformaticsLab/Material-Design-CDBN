%% crbm_inference
function [poshidexp2] = crbm_3rdlayer(image, patch, W, weight, Tlist, params, ii)
    rng(0);
    [H,L,numchannels] = size(image);
    
    ws = sqrt(size(W,1)/numchannels);
    numhid = size(W,2); % number of filters
    numtx = size(Tlist,1);
    batchsize = (H-ws+1)*(L-ws+1)*numchannels;
    numvis = ws*ws*numchannels;

    hidprob = zeros(numhid, numtx+1, batchsize/numchannels, 'single');
%     if params.optgpu,
%         hidprob = gpuArray(hidprob);
%     end
    
    % reshape for speedup
    vishid_up = W;
    vishid_down = W;
    if strcmp(params.intype, 'real'),
        vishid_up = 1/weight.sigma*vishid_up;
        vishid_down = weight.sigma*vishid_down;
    end
    hbiasmat = reshape(repmat(weight.hidbias(:,1:numtx), [1, 1, batchsize/numchannels]), numhid, numtx*batchsize/numchannels);
%     vbiasmat = repmat(weight.visbias, [1, batchsize]);
    
    xb = zeros(batchsize, ws^2);
    for i = 1 : (L-ws+1)
        for j = 1 : (H-ws+1)
            for k = 1:numchannels
                try
                    temp = image(j:ws+j-1,i:ws+i-1,k);
                    xb((j+(L-ws+1)*(i-1)-1)*numchannels+k,:) = reshape(temp,[1,ws*ws]); 
                catch
                    wait = 1;
                end
            end
        end    
    end
      
%     if params.optgpu,
%         xb = gpuArray(xb);
%     end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %Removed since no translation
%     A = zeros(size(Tlist{1}, 1)*params.numtx, size(Tlist{1}, 2), 'single');
%     if params.optgpu,
%         A = gpuArray(A);
%     end
% 
%     for i = 1:params.numtx,
%         if params.optgpu,
%             A((i-1)*size(Tlist{1}, 1)+1:i*size(Tlist{1}, 1), :) = gpuArray(single(full(Tlist{i})));
%         else
%             A((i-1)*size(Tlist{1}, 1)+1:i*size(Tlist{1}, 1), :) = full(Tlist{i});
%         end
%     end
%     Tlist_matrix = A; clear A;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % positive phase       
    Tx = reshape(xb', numvis, numtx*batchsize/numchannels);
    hidprob = reshape(hidprob, numhid, numtx+1, batchsize/numchannels);
    

% % for random reconstruct
%     hidstate_rand=zeros(numhid,numtx+1,size(Tx,2));
%     for k = 1:size(Tx,2)
%         for i = 1:numhid
%             temp = randi([1 numtx+1],1);
%             hidstate_rand(i,temp,k) = 1;
%         end
%     end
%     hidstate=hidstate_rand(:,1:numtx,:);
%     
    hidprob(:, 1:numtx, :) = reshape(vishid_up'*Tx + hbiasmat, numhid, numtx, batchsize/numchannels);
    hidprob(:, numtx+1, :) = 0;
    hidprob = exp(bsxfun(@minus, hidprob, max(hidprob, [], 2)));
    hidprob = bsxfun(@rdivide, hidprob, sum(hidprob, 2));
    
    % negative phase
    % hidden sampling
    hidprob = reshape(permute(hidprob, [2 1 3]), numtx+1, numhid*batchsize/numchannels);
    [hidstate, ~] = sample_multinomial(hidprob, params.optgpu);
    hidstate = permute(reshape(hidstate(1:numtx, :), numtx, numhid, batchsize/numchannels), [2 1 3]);
    fname = sprintf('hidstates45th_WB_(24f40f144f288f6ws9ws9ws12ws)_%d',ii);
    save(sprintf('3rd_hidstate/%s.mat',fname),'hidstate', '-v7.3');
%     

    negdata = zeros(L-ws+1, H-ws+1, numchannels);
    for nf = 1:numhid
        for nt = 1:numtx
            filter_t = vishid_down(:,nf);
            filter_t = reshape(filter_t,[ws,ws,numchannels]);
            S = reshape(hidstate(nf,nt,:),L-ws+1,H-ws+1);
            temp = conv2_mult(S, filter_t, 'same');
            negdata = negdata + temp;
        end
    end

            
    poshidexp2 = negdata;
    return
end
