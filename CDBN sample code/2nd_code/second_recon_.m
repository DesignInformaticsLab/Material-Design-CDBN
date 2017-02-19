addpath('results')
fname=sprintf('WB_2nd_10_(2f40f6ws9ws_nonorm)_alloy_w9_b40_trans_ntx1_gr1_pb0.1_pl20_iter_200');
load(sprintf('results/%s.mat',fname));
addpath('utils','function_code');
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
% numch=1;
% numchannels=params.numch;
weight=gpu2cpu_struct(weight);


W=weight.vishid;
% temp=zeros(24^2,8);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for i = 1:size(W,2)
%     temp(:,i)=W(1:24^2,i);
% end
% W=temp;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% W=reshape(W,[size(W,1),1,size(W,2)]);
% W=W(:,1,:);
hbias_vec=weight.hidbias;
dataname='alloy_scale';
% dataname='image';
Tlist = get_txmat(params.txtype, params.rs, params.ws, params.grid, params.numrot, params.numch);
params.numtx = length(Tlist);

for ii = 1:100
fname=sprintf('hidstates1st_WB_100_pooled_(2f12r6wsP10Pb01)_%d',ii);
load([fname '.mat'],'temp3');
fprintf('Loading negdata %d...\n',ii);
% hidstate=permute(hidstate,[3,2,1]);
hidstate=reshape(temp3,[97*97 24]);

image2=hidstate;
image2=reshape(image2,[sqrt(size(image2,1)),sqrt(size(image2,1)),size(image2,2)]);


image_reconstruct = crbm_inference_2nd(image2, patch, W,weight, Tlist, params,ii); % remove rbm1.pars and set the value0.2 inside the function 10/15/2015
end

