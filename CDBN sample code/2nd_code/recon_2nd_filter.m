addpath('C:\doiUsers\Ruijin\materialCDBM\ruijin\CRBM\Honglak Lee\rbm_rotation\secondlayer_training\function_code')
addpath('C:\doiUsers\Ruijin\materialCDBM\ruijin\CRBM\Honglak Lee\rbm_rotation\secondlayer_training\utils')


load('TEST4_filter_1layer_2f_6w.mat');
load('TEST4_Tlist_rot4_ws6_f2.mat');
load('2nd_TEST4_update2_2f4f_r4_6ws12ws_P10P30_alloy_w12_b04_trans_ntx1_gr1_pb0.1_pl30_iter_1000.mat')
numch = 8;
% filter_t=reshape(filter_t,[sqrt(size(filter_t,1)),sqrt(size(filter_t,1))]);
temp=gpu2cpu_struct(weight);
temp=temp.vishid;
filter_t=filters_t;
% filter_t=filter1_filters_test_6rot6filter;
filter_t_corr=zeros(size(Tlist_matrix,1),size(filter_t,2));
for i = 1:size(filter_t,2)
    filter_t_corr(:,i)=Tlist_matrix*filter_t(:,i);
%     figure(1),subplot(2,4,i),
end

% filter_t_corr=reshape(filter_t_corr,[size(filter_t,1),size(filter_t,2)*size(filter_t_corr,2)]);
filter_t_corr=reshape(filter_t_corr,[numel(filter_t_corr)/numch,numch]);

for j = 1:size(temp,2)
    filter_2nd_temp=temp(:,j);
% temp=reshape(temp,[24,24]);
% filter_2nd=temp;



% negdata = zeros(sqrt(size(filter_t,1)), sqrt(size(filter_t,1)));
negdata = zeros(17, 17);
for i=1:numch
%     for ii =1:size(filter_t,2)
        filter_t_temp=filter_t_corr(:,i);
        figure(10+j),subplot(2,4,i),display_network(filter_t_temp);
        filter_t_temp=reshape(filter_t_temp,[sqrt(size(filter_t_corr,1)),sqrt(size(filter_t_corr,1))]);
%         filter_t_temp=double(abs(im2bw(filter_t_temp,0.05)-1));
        filter_2nd = filter_2nd_temp((i-1)*size(temp,1)/numch+1:i*size(temp,1)/numch,:);
        figure(1+j),subplot(2,4,i),display_network(filter_2nd);
        filter_2nd=reshape(filter_2nd,[sqrt(size(temp,1)/numch),sqrt(size(temp,1)/numch)]);
% %         filter_t_temp=abs(filter_t_temp-1); %flip the filter, display 0.25~1
%         temp2=conv2(filter_t_temp,filter_2nd,'same');
        temp3=conv2(filter_2nd,filter_t_temp,'full');

%         figure(18),subplot(12,12,i),imshow(temp3,[-5 5])
        negdata = temp3+negdata;
%         negdata=sigmoid(negdata);
%         figure(5+10);subplot(10,10,i),imshow(negdata);
%         figure(j+10); subplot(1,8)imshow(negdata) negdata = temp2+negdata;
%     end
end
%     fname = sprintf('filter2nd_WB(8f48f)_ws24ws6_%d',j);
%     save(sprintf('%s.mat',fname),'negdata', '-v7.3');

% negdata=sigmoid(negdata);
figure(514); subplot(2,2,j),display_network(reshape(negdata,size(negdata,1)*size(negdata,2),1));
% figure(401); subplot(6,8,j),imshow(negdata);

% figure(j)
% display_network(reshape(negdata,size(negdata,1)*size(negdata,2),1));
% imshow(negdata);
end
