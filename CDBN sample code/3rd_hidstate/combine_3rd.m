for ii = 1:10
fname = sprintf('hidstates3rd_10_WB_(24f40f288f6ws9ws9ws)_%d',ii);
f1=load(sprintf('%s.mat', fname));   
temp = double([f1.hidstate;]);
temp = permute(temp,[3,2,1]);
xtr(ii,:) = temp(:)';
% f1=load([CIFAR_DIR '/filter8_ws12.mat']);
end
fname = sprintf('hidstates3rd_10_WB_(24f40f288f6ws9ws9ws)');
save(sprintf('3rd_hidstate/%s.mat',fname),'xtr', '-v7.3');
fprintf('data to be trained in the next layer is ----->\n 3rd_hidstate//%s.mat\n',fname);


% rearrange for max pooling
% for ii = 1:99
% fname = sprintf('hidstates_%d',ii);
% f1=load(sprintf('%s.mat', fname));   
% temp = double([f1.hidstate;]);
% for m = 1:size(temp,1)%6
%     for n = 1:size(temp,2)
%         temp2=temp(m,n,:);
%         temp2=reshape(temp2,[1,size(temp2,3)]);
%         temp3(m+(m-1)*n,:)=temp2;
%     end
% end
% % xtr{ii}=temp3;
% xtr(1+(ii-1)*36:1+(ii-1)*36+35,:) = temp3;
% % f1=load([CIFAR_DIR '/filter8_ws12.mat']);
% end
% % xtr=cell2mat(xtr);