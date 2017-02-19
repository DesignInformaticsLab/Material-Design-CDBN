addpath('results','alloy_mat','function_code','utils')
fname=sprintf('WB_100_f2rot12_alloy_w6_b02_rot_nrot12_pb0.1_pl20_iter_200');
load(sprintf('%s.mat',fname));
% addpath('../structure/');
load('WB.mat')
params.optgpu = 0;
spacing = 1;
ws=params.ws;
rs=params.rs;
kcd=params.kcd;
txtype = params.txtype;
grid = params.grid;
numrot=params.numrot;
% params.numrot = 1;
numch=params.numch;
% params.numch=36;
% numchannels=params.numch;
% weight=gpu2cpu_struct(weight);

W=gather(weight.vishid);
W=reshape(W,[size(W,1),1,size(W,2)]);
W=W(:,1,:);
hbias_vec=weight.hidbias;
dataname='alloy';
% dataname='image';
Tlist = get_txmat(params.txtype, params.rs, params.ws, params.grid, params.numrot, params.numch);
params.numtx = length(Tlist);

    
for ii = 1:100
% images_all = sample_images_all(dataname);

image = WB(ii,:);
image = reshape(image,[200,200]);
    
    [image_reconstruct]= crbm_inference(image, weight, Tlist, params,ii); % remove rbm1.pars and set the value0.2 inside the function 10/15/2015
%     figure,display_network(reshape(image_reconstruct,size(image_reconstruct,1)*size(image_reconstruct,2),1));
    store_test1(:,ii)=image_reconstruct(:);
end
