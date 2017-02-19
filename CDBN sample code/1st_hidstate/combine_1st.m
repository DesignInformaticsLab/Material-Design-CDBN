for ii = 1:100
fprintf('processing 1st hidstate %d...\n',ii);
fname = sprintf('hidstates1st_WB_100_(2f12r6wsP10Pb01)_%d',ii);
f1=load(sprintf('%s.mat', fname));   
temp = double([f1.hidstate;]);
temp = permute(temp,[3,1,2]);%% caution, changed from [3,2,1] 12/6 2016
temp = reshape(temp,[38025 24]);
for i = 1:24
    temp2=reshape(temp(:,i),[195 195]);
    temp2=imresize(temp2,[97 97]);
    temp3(:,i)=double(im2bw(temp2(:),0.2));
end
fname2 = sprintf('hidstates1st_WB_100_pooled_(2f12r6wsP10Pb01)_%d',ii);
save(sprintf('1st_hidstate/%s.mat',fname2),'temp3', '-v7.3');

xtr(ii,:) = temp3(:)';
% f1=load([CIFAR_DIR '/filter8_ws12.mat']);
end
fname3 = sprintf('hidstates1st_WB_100_pooled');
save(sprintf('1st_hidstate/%s.mat',fname3),'xtr', '-v7.3');
fprintf('data to be trained in the next layer is ----->\n 1st_hidstate//%s.mat\n',fname3);