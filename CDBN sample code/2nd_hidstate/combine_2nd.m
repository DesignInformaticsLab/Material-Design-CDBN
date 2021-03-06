for ii = 1:60
fprintf('processing 2nd hidstate %d...\n',ii);
fname = sprintf('hidstate_2nd_sandstone_(24f40f6ws9wsPb1010PL0301)_limcpatch_%d',ii);
f1=load(sprintf('%s.mat', fname));   
temp = double([f1.hidstate;]);
temp = permute(temp,[3,1,2]);%% caution, changed from [3,2,1] 12/9 2016
for i = 1:40
    temp2=reshape(temp(:,i),[89 89]);
    temp2=imresize(temp2,[44 44]);
    temp3(:,i)=double(im2bw(temp2(:)));
end
fname2 = sprintf('hidstates2nd_sandstone_pooled_(24f40f6ws9wsPb1010PL0301)_limcpatch_%d',ii);
save(sprintf('2nd_hidstate/%s.mat',fname2),'temp3', '-v7.3');

xtr(ii,:) = temp3(:)';
% f1=load([CIFAR_DIR '/filter8_ws12.mat']);
end
fname3 = sprintf('hidstates2nd_sandstone_pooled_(24f40f6ws9wsPb1010PL0301)_limcpatch');
save(sprintf('2nd_hidstate/%s.mat',fname3),'xtr', '-v7.3');
fprintf('data to be trained in the next layer is ----->\n 2nd_hidstate//%s.mat\n',fname3);