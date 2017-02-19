% clear;

%% reconstruction process from 3rd layer
addpath('utils','function_code','hidstate_3rdlayer_p2p2_(2f40f144f6ws9ws9ws12rP20P10P10Pb01)')
% One=load('WB_real_alloy_w24_b02_rot_nrot24_pb0.1_pl0.1_iter_500.mat');
One=load('WB_nowh_P20Pb01_rot12_2f_6ws_alloy_w6_b02_rot_nrot12_pb0.1_pl20_iter_4000.mat');

fname=sprintf('WB_2nd_pool2_hidstate_(2f40f6ws9ws12rP20P20Pb01)_alloy_w9_b40_trans_ntx1_gr1_pb0.1_pl20_iter_2000');
Two=load(sprintf('%s.mat',fname));

fname=sprintf('3rd_POOL2(imresize)_(real_bibi)_(2f40f288f6ws9ws9ws12rP20P10P10Pb01)_alloy_w9_b288_trans_ntx1_gr1_pb0.1_pl10_iter_1000');
Three=load(sprintf('%s.mat',fname));

%% 3rd layer filter pooling back
weight3=Three.weight;
W_Three=gather(weight3.vishid);

% for i = 1:size(W_Three,2)
%     W_temp=reshape(W_Three(:,i),[size(W_Three,1)/40 40]);
%     for j = 1:40
%         W_temp2=reshape(W_temp(:,j),[sqrt(size(W_temp,1)) sqrt(size(W_temp,1))]);
%         W_temp2=imresize(W_temp2,[sqrt(size(W_temp,1))*2 sqrt(size(W_temp,1))*2]);
%         W_temp3(:,j)=W_temp2(:);
%     end
%     W_Three_Pool(:,i)=W_temp3(:);
% end
% clear W_temp3;clear W_temp2;clear W_temp;
% W_Three=W_Three_Pool;

params_Three=Three.params;
ws_Three=params_Three.ws;


%% parameter initialize
temp=One.weight;
temp=gpu2cpu_struct(temp);
W_one=temp.vishid;
weight=One.weight;
clear temp;
temp=Two.weight;
temp=gpu2cpu_struct(temp);
W_two=temp.vishid;

%% 2nd layer filter pooling back
% for i = 1:size(W_two,2)
%     W_temp=reshape(W_two(:,i),[9*9 24]);
%     for j = 1:24
%         W_temp2=reshape(W_temp(:,j),[9 9]);
%         W_temp2=imresize(W_temp2,[18 18]);
%         W_temp3(:,j)=W_temp2(:);
%     end
%     W_two_Pool(:,i)=W_temp3(:);
% end
W_two_Pool=W_two;

% dimension of the reconstruction image
L3=36;H3=36;

params_One=One.params;
params_Two=Two.params;
ws_Two=params_Two.ws;
Tlist = get_txmat(params_One.txtype, params_One.rs, params_One.ws, params_One.grid, params_One.numrot, params_One.numch);

%% reconstruction process
for ii = 1:100

%%%%%%initialize store size(could be erased)%%%
    hidstate_2nd_layer=zeros(24,20736);
    hidstate_pool2=zeros(72,72,40);
%               
%     hidstate=W(:,ii);
%     hidstate=reshape(hidstate,[36*36 144])';
%     hidstate=reshape(hidstate,[144 1 36*36]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    
%     hidstate=reshape(double(im2bw(reconst(:,ii),0.1)),[186624/144 144])';
%     hidstate=reshape(im2bw(recon_4to3_rand(:,ii),0.1),[1296 288])';

    hidstate=reshape(im2bw(hidstate_sim_f(:,ii)),[288 373248/288]);
%     hidstate=reshape(double(im2bw(reconst_4to3(:,ii),0.1)),[373248/288 288])';
    
%     hidstate=reshape(im2bw(reconst_ori(:,ii)),[1296 288])';
    hidstate=double(reshape(hidstate,[288 1 1296]));
    
%     hidstate=hidstate_sim_f(:,ii);
%     hidstate=reshape(hidstate,[144 186624/144]);
%     hidstate=reshape(hidstate,[144 1 1296]);
    
%     fname=sprintf('hidstates3nd_WB_nowh(p2p2)_imresize_(2f40f144f6ws9ws9ws12rP20P10P10Pb01)_%d',ii); %WB
%     load([fname '.mat'],'hidstate');
    %% 3rd layer hidstate pool back by2
%     hidstate=reshape(hidstate,[size(hidstate,1) size(hidstate,3)]);
%     for pp=1:size(hidstate,1)
%         hidstate_temp=reshape(hidstate(pp,:),[sqrt(size(hidstate,2)),sqrt(size(hidstate,2))]);
%         hidstate_pool2 = imresize(hidstate_temp,[sqrt(size(hidstate,2))*2,sqrt(size(hidstate,2))*2]);
%         hidstate_pool2 = double(im2bw(hidstate_pool2));
%         
%         hidstate_3rd_layer(pp,:)=hidstate_pool2(:);
%     end
%     hidstate=hidstate_3rd_layer;
%     hidstate=reshape(hidstate,[288 1 5184]);
   %% 3rd layer reconstruction (pool back by 2)
    numchannels=params_Three.numch;
    negdata = zeros(L3, H3, numchannels);
    for nf = 1:params_Three.numhid
        for nt = 1:params_Three.numtx
            filter_t = W_Three(:,nf);
            filter_t = reshape(filter_t,[ws_Three,ws_Three,numchannels]);
            S = reshape(hidstate(nf,nt,:),L3,H3);
            temp = conv2_mult(S, filter_t, 'same');
            negdata = negdata + temp;
        end
    end
    %% set bw threshold to set up the hidstate(or apply fdog)
%     negdata_temp=(negdata-min(negdata(:)))/max(max(max((negdata-min(negdata(:))))));
%     negdata_temp=double(im2bw(reshape(negdata_temp,[36*36 40]),0.6));
%     hidstate=negdata_temp;
 
%     load('hidstate_fdog2'); %this step is by FDoG
%     hidstate=hidstate_fdog2';
%     hidstate=reshape(hidstate,[36 36 40]);
    
%     negdata = sigmoid(negdata);
%     hidstate=negdata;
    temp_binary=zeros(36,36,40);
    [svals,idx]=sort(negdata(:),'descend');
    thre=svals(round(36*36*40*0.1));
    for aa = 1:size(negdata,3)
        temp_fdog2=reshape(negdata(:,:,aa),[36 36]);
%         [Gmag,Gdir] = imgradient(temp_fdog2);
%         temp_fdog2=temp_fdog2>thre;
%         temp_fdog2(Gmag>300)=1;
        temp_binary(:,:,aa)=temp_fdog2;
    end
    hidstate=temp_binary;
    %% 3rd layer hidstate pool back by2
    for pp = 1:size(hidstate,3)
        hidstate_pool=imresize(hidstate(:,:,pp),2);
        hidstate_pool=double(im2bw(hidstate_pool));
        hidstate_pool2(:,:,pp)=hidstate_pool;
    end
    hidstate=hidstate_pool2;
    
    hidstate=permute(hidstate,[3,1,2]);
    hidstate=reshape(hidstate,[1,40,(L3*2)^2]);
    
        
%% 2nd layer reconstruction (pooled by 2)   
vishid_up = W_two_Pool;
vishid_down_two = W_two_Pool;
if strcmp(params_Two.intype, 'real'),
    vishid_up = 1/weight.sigma*vishid_up;
    vishid_down_two = weight.sigma*vishid_down_two;
end
numchannels=params_Two.numch;
vishid_down_two=gather(vishid_down_two);
negdata = zeros(L3*2, H3*2, numchannels);
    for nf = 1:params_Two.numhid
        for nt = 1:params_Two.numtx
            filter_t = vishid_down_two(:,nf);
            filter_t = reshape(filter_t,[ws_Two,ws_Two,numchannels]);
            S = reshape(hidstate(nt,nf,:),L3*2,H3*2);
            S1 = S;
            S1=im2bw(S1);
            S1=double(S1);
            filter_x = ones(2,2);
            for i = 1:1
                S1=conv2(S1,filter_x,'same');
                S1=floor(S1/4);
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
    
    [svals,idx]=sort(negdata(:),'descend');
    thre=svals(round(72*72*24*0.1));
    temp_binary=zeros(72,72,24);
    for aa = 1:size(negdata,3)
        temp_fdog2=reshape(negdata(:,:,aa),[72 72]);
%         temp_binary(:,:,aa)=temp_fdog2>thre;        
%         [Gmag,Gdir] = imgradient(temp_fdog2);
%         temp_fdog2=temp_fdog2>thre;
%         temp_fdog2(Gmag>800)=1;
        temp_binary(:,:,aa)=temp_fdog2;
        
    end
    hidstate=temp_binary;    
    
%     negdata_temp=(negdata-min(negdata(:)))/max(max(max((negdata-min(negdata(:))))));
%     negdata_temp=double(im2bw(reshape(negdata_temp,[72*72 24]),0.65));
%     hidstate=negdata_temp;
%     hidstate=reshape(hidstate,[72 72 24]);
    
% negdata = sigmoid(negdata);
% negdata=gather(negdata);
% hidstate=negdata;

%% 2nd layer hidstate pool back by 2
    for pp=1:size(hidstate,3)
        hidstate_temp=hidstate(:,:,pp);
%         for m = 1:size(hidstate,2)
%             for n = 1:size(hidstate,1)
%                 if hidstate_temp(m,n)==1
%                     hidstate_pool2(2*m-1:2*m,2*n-1:2*n)=1;
%                 else
%                     hidstate_pool2(2*m-1:2*m,2*n-1:2*n)=0;
%                 end
%             end
%         end
        hidstate_pool2=imresize(hidstate_temp,[size(hidstate,1)*2,size(hidstate,2)*2]);
        hidstate_2nd_layer(pp,:)=hidstate_pool2(:);
    end
    hidstate=hidstate_2nd_layer';
    hidstate=reshape(hidstate,[144 144 24]);


hidstate=permute(hidstate,[3,1,2]);
hidstate=reshape(hidstate,[12,2,(L3*4)^2]);    



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
negdata = zeros(L3*4, H3*4, numchannels);
    for nf = 1:numhid
        for nt = 1:numtx
            filter_t = full(Tlist{nt})'*vishid_down_one(:,nf);
            filter_t = reshape(filter_t,[ws_One,ws_One,numchannels]);
            S = reshape(hidstate(nt,nf,:),L3*4,H3*4);
            S1 = S;
            S1=im2bw(S1);
            S1=double(S1);
            filter_x = ones(2,2);
            for i = 1:1
                S1=conv2(S1,filter_x,'same');
                S1=floor(S1/4);
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
% negdata = sigmoid(negdata);
negdata=gather(negdata);

%     fname = sprintf('recon_2nd_%d',ii);
%     save(sprintf('%s.txt',fname),'negdata', '-ascii');
    
% figure(ii);display_network(reshape(negdata(:,:,1),size(negdata,1)*size(negdata,2),1),ii);
% figure(112);subplot(10,10,ii),imshow(negdata);

% figure(162);display_network(reshape(negdata(:,:,1),size(negdata,1)*size(negdata,2),1));
store_total4(:,ii)=negdata(:);

end