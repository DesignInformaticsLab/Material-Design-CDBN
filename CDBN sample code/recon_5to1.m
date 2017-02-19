clear
addpath('results','5th_hidstate')
W5_temp=load('alloy2_5th(rbm)_(2f40f288f1000f6ws9ws9ws36wsP10Pb0401)_alloy_w1_b30_trans_ntx1_gr1_pb0_pl0_iter_20000.mat');
W4_temp=load('alloy2_4th(rbm)_(24f40f288f1000f6ws9ws9ws36ws)_alloy_w36_b1000_trans_ntx1_gr1_pb0.1_pl10_iter_2000.mat');

W5=gather(W5_temp.weight.vishid);
vbias5=gather(W5_temp.weight.visbias);
W4=gather(W4_temp.weight.vishid);
vbias4=gather(W4_temp.weight.visbias);
k=0;
%% original reconstruction
for ii = 1:100
    fprintf('loading sample %d\n',ii);
%     fname=sprintf('hidstates5th_10_WB_(24f40f288f1000f30f6ws9ws9ws36ws1ws)_%d',ii); %WB
%     load([fname '.mat'],'hidstate');
%     hidstate=abs(1-hidstate);% reverse

    hidstate=randi(100,[30,1]);
    hidstate(hidstate>40)=1;
    hidstate(hidstate~=1)=0;
    
%     hidstate=zeros(30,1);
%     hidstate(1,1)=1;
    
    hid_4th=sigmoid(W5*hidstate+vbias5);
    hid_3rd=sigmoid(W4*hid_4th+vbias4);
    hid_3rd_store(:,ii)=double(im2bw(hid_3rd(:),0.5)); %threshold
    
    for i = 1:288
        hid_temp=reshape(hid_3rd_store(:,ii),[1296 288])';
        temp=hid_temp(i,:);
        if sum(temp(:))>0.1*sum(hid_temp(:))
            temp(:)=0;
            k=k+1;
        end
        hidstate_sim(i,:)=temp;
    end
    hid_3rd_f(:,ii)=hidstate_sim(:);
end
