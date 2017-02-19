clear;
addpath('function_code','utils','visualization_recon');

load('filter_2nd_layer.mat');
load('WB_2nd_10_(2f40f6ws9ws_nonorm)_alloy_w9_b40_trans_ntx1_gr1_pb0.1_pl20_iter_2000.mat')
numch = 24;

%define 3rd layer filter
W_Three=gather(weight.vishid);
%pool back 3rd layer filter
for i = 1:size(W_Three,2)
    W_temp=reshape(W_Three(:,i),[9*9 24]);
    for j = 1:numch
        W_temp2=reshape(W_temp(:,j),[9 9]);
        W_temp2=imresize(W_temp2,[9*4 9*4]);
        W_temp3(:,j)=W_temp2(:);
    end
    W_Three_Pool(:,i)=W_temp3(:);
end
clear W_temp3;clear W_temp2;clear W_temp;
W_Three=W_Three_Pool;


% define 2nd layer filter
filter_t_corr=filter_2nd_layer';

filter_t_corr=reshape(filter_t_corr,[numel(filter_t_corr)/numch,numch]);
for j = 1:size(W_Three,2)
    filter_3rd_temp=W_Three(:,j);

negdata = zeros(9*4, 9*4);
for i=1:numch
%     for ii =1:size(filter_t,2)
        filter_t_temp=filter_t_corr(:,i);
        filter_t_temp=reshape(filter_t_temp,[sqrt(size(filter_t_corr,1)),sqrt(size(filter_t_corr,1))]);
        filter_3rd = filter_3rd_temp((i-1)*size(W_Three,1)/numch+1:i*size(W_Three,1)/numch,:);
        filter_3rd=reshape(filter_3rd,[sqrt(size(W_Three,1)/numch),sqrt(size(W_Three,1)/numch)]);
%         filter_t_temp=abs(filter_t_temp-1); %flip the filter, display 0.25~1
%         temp2=conv2(filter_t_temp,filter_3rd,'same');
        temp3=conv2(filter_3rd,filter_t_temp,'same');
%         figure(18),subplot(12,12,i),imshow(temp3,[-5 5])
        negdata = temp3+negdata;
%         negdata=sigmoid(negdata);
%         figure(5+10);subplot(10,10,i),imshow(negdata);
%         figure(j+10); subplot(1,8)imshow(negdata) negdata = temp2+negdata;
%     end
end
store(:,j)=negdata(:);
% figure(104); subplot(12,12,j),display_network(reshape(negdata,size(negdata,1)*size(negdata,2),1));

end
