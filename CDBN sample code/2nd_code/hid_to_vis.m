% One=load('WB_real_alloy_w24_b02_rot_nrot24_pb0.1_pl0.1_iter_500.mat');
One=load('WB_f2rot12_alloy_w6_b02_rot_nrot12_pb0.4_pl10_iter_2000.mat');
addpath('utils','function_code')
addpath('2nd_hidstate')
fname=sprintf('WB_2nd_hidstate_(2f40f6ws9ws)_alloy_w9_b40_trans_ntx1_gr1_pb0.1_pl10_iter_2000');
% fname=sprintf('2nd_WB(real_bi)_f96(f12)_plamda005_ws12_alloy_w12_b96_trans_ntx1_gr1_pb0.1_pl0.05_iter_2000');
Two=load(sprintf('%s.mat',fname));

%% filter pooling back
W=gather(Two.weight);
W=gather(W.vishid);
for i = 1:size(W,2)
    W_temp=reshape(W(:,i),[9*9 24]);
    for j = 1:24
        W_temp2=reshape(W_temp(:,j),[9 9]);
        W_temp2=imresize(W_temp2,[18 18]);
        W_temp3(:,j)=W_temp2(:);
    end
    W_two(:,i)=W_temp3(:);
end


%% parameter initialize
temp=One.weight;
temp=gpu2cpu_struct(temp);
W_one=temp.vishid;
weight=One.weight;
clear temp;
vishid_down=W_two;

L2=195;H2=195;L1=178;H1=178;
% L2=89;H2=89;L1=78;H1=78;

params_One=One.params;
params_Two=Two.params;
ws_Two=params_Two.ws;
Tlist = get_txmat(params_One.txtype, params_One.rs, params_One.ws, params_One.grid, params_One.numrot, params_One.numch);

negdata_2nd=zeros(100,178*178);
for ii = 1:100
    
    fname=sprintf('hidstate_2nd_WB_(2f40f6ws9ws)_%d',ii); %WB
    load([fname '.mat'],'hidstate');
    
    %% pool back hidstate
    for pp=1:size(hidstate,1)
        hidstate=reshape(hidstate,[40 7921]);
        hidstate_temp=reshape(hidstate(pp,:),[sqrt(size(hidstate,2)),sqrt(size(hidstate,2))]);
       for m = 1:sqrt(size(hidstate,2))
            for n = 1:sqrt(size(hidstate,2))
                if hidstate_temp(m,n)==1
                    hidstate_pool2(2*m-1:2*m,2*n-1:2*n)=1;
                else
                    hidstate_pool2(2*m-1:2*m,2*n-1:2*n)=0;
                end
            end
       end
        hidstate_2nd_layer(pp,:)=hidstate_pool2(:);
    end
    hidstate=reshape(hidstate_2nd_layer,[40 1 178*178]); 

        
        
        
    numchannels=params_Two.numch;
    negdata = zeros(L2-ws_Two*2+1, H2-ws_Two*2+1, numchannels);
    for nf = 1:params_Two.numhid
        for nt = 1:params_Two.numtx
            filter_t = vishid_down(:,nf);
            filter_t = reshape(filter_t,[ws_Two*2,ws_Two*2,numchannels]);
            S = reshape(hidstate(nf,nt,:),L2-ws_Two*2+1,H2-ws_Two*2+1);
            temp = conv2_mult(S, filter_t, 'same');
            negdata = negdata + temp;
        end
    end
negdata = sigmoid(negdata);
hidstate=negdata;
hidstate=permute(hidstate,[3,1,2]);
hidstate=reshape(hidstate,[12,2,L1^2]);



L=sqrt(size(hidstate,3));H=L;
numchannels=params_One.numch;
numhid=params_One.numhid;
numtx=params_One.numtx;
ws_One=params_One.ws;

vishid_up = W_one;
vishid_down_one = W_one;
if strcmp(params_One.intype, 'real'),
    vishid_up = 1/weight.sigma*vishid_up;
    vishid_down_one = weight.sigma*vishid_down_one;
end

vishid_down_one=gather(vishid_down_one);
negdata = zeros(L1, H1, numchannels);
    for nf = 1:numhid
        for nt = 1:numtx
            filter_t = full(Tlist{nt})'*vishid_down_one(:,nf);
            filter_t = reshape(filter_t,[ws_One,ws_One,numchannels]);
            S = reshape(hidstate(nt,nf,:),L1,H1);
            S1 = S;
            S1=im2bw(S1);
            S1=double(S1);
            filter_x = ones(2,2);
            for i = 1:1
                S1=conv2(S1,filter_x,'same');
                S1=floor(S1/2);
            end
            S=S1;
            S=double(S);
            filter_t=gather(filter_t);
            temp = conv2_mult(S, filter_t, 'same');
            temp=gather(temp);
            negdata = negdata + temp;
        end
%         negdata = sigmoid(negdata);
    end
negdata = sigmoid(negdata);
negdata=gather(negdata);
negdata_2nd(ii,:)=negdata(:);
%     fname = sprintf('recon_2nd_(2f40f6ws18ws12rP20P10Pb01)_%d',ii);
%     save(sprintf('%s.txt',fname),'negdata', '-ascii');
    
figure,display_network(reshape(negdata(:,:,1),size(negdata,1)*size(negdata,2),1),ii);

% figure(112);subplot(10,10,ii),imshow(negdata);
% figure(2);display_network(reshape(negdata(:,:,1),size(negdata,1)*size(negdata,2),1));
end