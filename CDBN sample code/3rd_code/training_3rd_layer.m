% -----------------------------------------------------------------------
%   Train transformation-invariant restricted Boltzmann machines
%
%   xtr         : training data
%   params      : parameters
%   Tlist       : list of transformation matrices (cell)
%
%   [Learning Invariant Representations with Local Transformations
%    Kihyuk Sohn and Honglak Lee. ICML 2012]
% -----------------------------------------------------------------------

% function [weight, params, history] = training_2nd_layer(xtr, params, Tlist)
function [weight, params, history] = training_3rd_layer(xtr, params)
rng('default');

% if ~exist(sprintf('%s_iter_%d.mat', params.fname,params.maxiter-1500))
    
% weight initialization
weight.vishid = 0.02*randn(params.numvis, params.numhid);
weight.visbias = zeros(params.rSize, 1);
weight.hidbias = zeros(params.numhid, params.numtx);

% initialize sigma0 from Kmeans
if strcmp(params.intype, 'real'),
    [label, center] = litekmeans(double(xtr)', params.numhid, true, 100);
    center = center';
    weight.sigma = sqrt(mean(mean( (xtr - center(:, label)).^2, 2)));
    clear center label;
end

if params.optgpu,
    cpu2gpu_struct(weight);
end

% filename to save
if ~isfield(params, 'fname'),
    if strcmp(params.txtype,'rot'),
        fname = sprintf('trbm_2nd_layer_Filter12_ws24_%s_w%d_b%02d_%s_nrot%d_pb%g_pl%g', ...
            params.dataset, params.ws, params.numhid, params.txtype, params.numtx, params.pbias, params.plambda);
    elseif strcmp(params.txtype,'trans') || strcmp(params.txtype,'scale'),
        fname = sprintf('trbm_2nd_layer_Filter12_ws24_%s_w%d_b%02d_%s_ntx%d_gr%d_pb%g_pl%g', ...
            params.dataset, params.ws, params.numhid, params.txtype, params.numtx, params.grid, params.pbias, params.plambda);
    end
    params.fname  = sprintf('%s/%s', params.savepath, fname);
end



% structs for gradients
grad = replicate_struct(weight, 0);
pos = replicate_struct(weight, 0);
neg = replicate_struct(weight, 0);

% momentum
initialmomentum  = 0.01; % change from 0.1 to 0.01
finalmomentum    = 0.9;

% convert Tlist (cell) into matrix
% A = zeros(size(Tlist{1}, 1)*params.numtx, size(Tlist{1}, 2), 'single');
% if params.optgpu,
%     A = gpuArray(A);
% end
% 
% for i = 1:params.numtx,
%     if params.optgpu,
%         A((i-1)*size(Tlist{1}, 1)+1:i*size(Tlist{1}, 1), :) = gpuArray(single(full(Tlist{i})));
%     else
%         A((i-1)*size(Tlist{1}, 1)+1:i*size(Tlist{1}, 1), :) = full(Tlist{i});
%     end
% end
% Tlist = A; clear A;


% -----------------------------------------------------------------------
%                                                           learning
% -----------------------------------------------------------------------

nbatch = min(floor(size(xtr, 2)/params.batchsize), floor(100000/params.batchsize));
numvis = params.numvis;
numhid = params.numhid;
numtx = params.numtx;
batchsize = params.batchsize;

history.error = zeros(params.maxiter, 1);
history.sparsity = zeros(params.maxiter, 1);
history.hidbias = zeros(params.maxiter, 1);
history.visbias = zeros(params.maxiter, 1);
history.vishid = zeros(params.maxiter, 1);

hidprob = zeros(numhid, numtx+1, batchsize, 'single');
if params.optgpu,
    hidprob = gpuArray(hidprob);
end
temp_sparsitycheck=0;
for t = 1:params.maxiter,

    % momentum update
    if t < 100,
        momentum = initialmomentum;
    else
        momentum = finalmomentum;
    end
    lr = params.epsilon/(1+params.epsdecay*t);
    
    t20s = tic;
    
    % monitoring variables
    rec_err_epoch = zeros(nbatch, 1);
    sparsity_epoch = zeros(nbatch, 1);
    tetot = zeros(nbatch, 1);
    
    imidx_batch = randperm(size(xtr, 2));
    if mod(t,1000)==0
        aaa=1;
    end
    for i = 1:nbatch,
        % generate random patch
        batchidx = imidx_batch((i-1)*params.batchsize+1:i*params.batchsize);
        xb = xtr(:, batchidx);
        
        if params.optgpu,
            xb = gpuArray(xb);
        end
        
        % update trbm
        teptot = 0;
        tsp = tic;
        
        % reshape for speedup
        vishid_up = weight.vishid;
        vishid_down = weight.vishid;
        if strcmp(params.intype, 'real'),
            vishid_up = 1/weight.sigma*vishid_up;
            vishid_down = weight.sigma*vishid_down;
        end
        hbiasmat = reshape(repmat(weight.hidbias, [1, 1, batchsize]), numhid, numtx*batchsize);
        vbiasmat = repmat(weight.visbias, [1, batchsize]);
        
        % positive phase       
%         Tx = reshape(Tlist*xb, numvis, numtx*batchsize);
        Tx = reshape(xb, numvis, numtx*batchsize);
        hidprob = reshape(hidprob, numhid, numtx+1, batchsize);
        hidprob(:, 1:numtx, :) = reshape(vishid_up'*Tx + hbiasmat, numhid, numtx, batchsize);
        hidprob(:, numtx+1, :) = 0;
        hidprob = exp(bsxfun(@minus, hidprob, max(hidprob, [], 2)));
        hidprob = bsxfun(@rdivide, hidprob, sum(hidprob, 2));
        
        % gradient
        hidprob_rs = reshape(hidprob(:, 1:numtx, :), numhid, numtx*batchsize);
        pos.vishid = Tx*hidprob_rs'/batchsize - params.l2reg*weight.vishid;
        pos.hidbias = squeeze(sum(hidprob(:, 1:numtx, :), 3))/batchsize;
        pos.visbias = sum(xb, 2)/batchsize;
        
        % gradient w.r.t. sparsity
        hmh = hidprob_rs.*(1-hidprob_rs);
        dh_reg = (params.pbias - sum(pos.hidbias, 2)).*sum(hmh, 2)/batchsize;
        tmp = repmat(params.pbias - sum(pos.hidbias, 2), [1, numtx*batchsize]);
        dvh_reg = Tx*(hmh.*tmp)'/batchsize;
        
        pos.vishid = pos.vishid + params.plambda*dvh_reg;
        pos.hidbias = bsxfun(@plus, pos.hidbias, params.plambda*dh_reg);
        
        
        % reconstruction
%         reconst = Tlist'*reshape(vishid_down*hidprob_rs, numvis*numtx, batchsize);
        reconst = reshape(vishid_down*hidprob_rs, numvis*numtx, batchsize);
        reconst = reconst + vbiasmat;
%         reconst=gather(reconst); %12/08/2015 out of gpu memory
        if strcmp(params.intype, 'binary'),
            reconst = sigmoid(reconst);
        end
        
        % error and sparsity
%         xb=gather(xb);reconst=gather(reconst); %out of gpu memory
        rec_err = sum((xb(:) - reconst(:)).^2)/(batchsize*params.rSize);
        sparsity = sum(sum(sum(hidprob(:, 1:numtx, :))))/(batchsize*numhid);
        
        % monitoring variables
        rec_err_epoch(i) = gather(rec_err);
        sparsity_epoch(i) = gather(sparsity);
        
        % negative phase
        % hidden sampling
        hidprob = reshape(permute(hidprob, [2 1 3]), numtx+1, numhid*batchsize);
        [hidstate, ~] = sample_multinomial(hidprob, params.optgpu);
        hidstate = permute(reshape(hidstate(1:numtx, :), numtx, numhid, batchsize), [2 1 3]);
        
        for kcd = 1:params.kcd,
            % visible inference
            hidstate_rs = reshape(hidstate, numhid, numtx*batchsize);
%             negdata = Tlist'*reshape(vishid_down*hidstate_rs, numvis*numtx, batchsize);
            negdata = reshape(vishid_down*hidstate_rs, numvis*numtx, batchsize);
            negdata = negdata + vbiasmat;
%             negdata=gather(negdata); % gpu out of memory
            if strcmp(params.intype, 'binary'),
                negdata = sigmoid(negdata);
            end
            
            % hidden inference
%             Tx = reshape(Tlist*negdata, numvis, numtx*batchsize);
            Tx = reshape(negdata, numvis, numtx*batchsize);
            hidprob = reshape(hidprob, numhid, numtx+1, batchsize);
            hidprob(:, 1:numtx, :) = reshape(vishid_up'*Tx + hbiasmat, numhid, numtx, batchsize);
            hidprob(:, numtx+1, :) = 0;
            hidprob = exp(bsxfun(@minus, hidprob, max(hidprob, [], 2)));
            hidprob = bsxfun(@rdivide, hidprob, sum(hidprob, 2));
            
            if kcd == params.kcd,
                % compute gradient
                hidprob_rs = reshape(hidprob(:, 1:numtx, :), numhid, numtx*batchsize);
                neg.vishid = Tx*hidprob_rs'/batchsize;
                neg.hidbias = squeeze(sum(hidprob(:, 1:numtx, :), 3))/batchsize;
                neg.visbias = sum(negdata, 2)/batchsize;
            else
                % hidden sampling
                hidprob = reshape(permute(hidprob, [2 1 3]), numtx+1, numhid*batchsize);
                [hidstate, ~] = sample_multinomial(hidprob, params.optgpu);
                hidstate = permute(reshape(hidstate(1:numtx, :), numtx, numhid, batchsize), [2 1 3]);
            end
        end
        
        % time
        tep = toc(tsp);
        teptot = teptot + tep;
        tetot(i) = teptot;
        
        % update parameters
        [weight, grad] = update_params(weight, grad, pos, neg, momentum, lr);
        
        
        % print every 20 inner iterations
        if mod(i, 5) == 0,% changed from 20 to 5,10/28/2015
            mean_err = mean(rec_err_epoch(i-4:i));% changed from 19 to 4,10/28/2015
            mean_sparsity = mean(sparsity_epoch(i-4:i));% changed from 19 to 4,10/28/2015
            
            fprintf('epoch:%d, iteration %d/%d, err= %g, sparsity= %g mean(hbias)= %g,  mean(vbias)= %g\n', ...
                t, i, nbatch, mean_err, mean_sparsity, gather(mean(weight.hidbias(:))), gather(mean(weight.visbias(:))));
        end
    end
    
    % update sigma using reconstruction error
    history.error(t) = gather(mean(rec_err_epoch));
    history.sparsity(t) = gather(mean(sparsity_epoch));
    history.hidbias(t) = gather(mean(weight.hidbias(:)));
    history.visbias(t) = gather(mean(weight.visbias(:)));
    history.vishid(t) = gather(sqrt(sum(weight.vishid(:).^2)/params.numhid));
    if strcmp(params.intype, 'real'),
        eta_sigma = params.eta_sigma;
        weight.sigma = weight.sigma*(1-eta_sigma) + eta_sigma*sqrt(sum(rec_err_epoch)/nbatch);
    end
    fprintf('epoch %d error = %g \tsparsity_hid = %g\n', t, mean(rec_err_epoch), mean(sparsity_epoch));
    
%     control epsilon value
%     if mod(t,10) == 0
%         temp_sparsitycheck=history.sparsity(t) - history.sparsity(t-9);
%          fprintf('Check point epsilon %d epoch %d error = %g \tsparsity_hid = %g\n', params.epsilon, t, mean(rec_err_epoch), history.sparsity(t));
%     end
%     if t>10 && mod(t, 10) == 0 && temp_sparsitycheck>0
%         params.epsilon = params.epsilon/2;
%     end
%     if params.epsilon <0.00005
%        return 
%     end    
    if mod(t,1000) == 0,
        % save trained weight
        weight = cpu2gpu_struct(weight);
        history = cpu2gpu_struct(history);
        save([params.fname '_iter_' num2str(t) '.mat'], 'params', 't', 'history', 'weight');
        
        % convert back to gpu
        if params.optgpu,
            weight = gpu2cpu_struct(weight);
            history = cpu2gpu_struct(history);
        end
    end
    
%     if mod(t,500) == 0
%        addpath('C:\doiUsers\Ruijin\materialCDBM\ruijin\CRBM\Honglak Lee\rbm_rotation\secondlayer_training\results');
%        show the image after every 10 iteration
%        load(sprintf('%s_iter_%d.mat', params.fname,t))
%        temp=gpu2cpu_struct(weight);
%        temp2=temp.vishid;
%        figure(t);display_network(temp2)
%     end    
end
% end
weight = cpu2gpu_struct(weight);
history = cpu2gpu_struct(history);
save([params.fname '_iter_' num2str(t) '.mat'], 'params', 't', 'history', 'weight');

return;
