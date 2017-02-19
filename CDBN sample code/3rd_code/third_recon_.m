addpath('function_code','utils','results','2nd_hidstate')
fname=sprintf('WB_45th_(2f40f144f288f6w9ws9ws12ws)_alloy_w12_b288_trans_ntx1_gr1_pb0.1_pl10_iter_2000');
load(sprintf('%s.mat',fname));


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

hbias_vec=weight.hidbias;
dataname='alloy_scale';
% dataname='image';
Tlist = get_txmat(params.txtype, params.rs, params.ws, params.grid, params.numrot, params.numch);
params.numtx = length(Tlist);

for ii = 1:100
fname=sprintf('hidstates3nd_WB_nowh(p2p2)_imresize_(2f40f144f6ws9ws9ws12rP20P10P10Pb01)_%d',ii);
load([fname '.mat'],'hidstate')

image2=hidstate;
image2=permute(image2,[3 1 2]); %special for 4.5 layer training
image2=reshape(image2,[sqrt(size(image2,1)) sqrt(size(image2,1)) size(image2,2)]);

image_reconstruct = crbm_3rdlayer(image2, patch, W,weight, Tlist, params,ii); % remove rbm1.pars and set the value0.2 inside the function 10/15/2015
end

