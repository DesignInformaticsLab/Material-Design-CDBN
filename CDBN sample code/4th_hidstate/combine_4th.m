for ii = 1:10
fname = sprintf('hidstates4th_10_WB_(24f40f288f1000f6ws9ws9ws36ws)_%d',ii);
f1=load(sprintf('%s.mat', fname));   
temp = double([f1.hidstate;]);
xtr(ii,:) = temp(:)';
% f1=load([CIFAR_DIR '/filter8_ws12.mat']);
end
fname = sprintf('hidstates4th_10_WB_(24f40f288f1000f6ws9ws9ws36ws)');
save(sprintf('4th_hidstate/%s.mat',fname),'xtr', '-v7.3');