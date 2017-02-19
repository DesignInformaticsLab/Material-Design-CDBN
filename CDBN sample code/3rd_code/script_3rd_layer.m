addpath('utils','function_code')
acc = {};
fname = {};
optgpu = 1;
if ~exist('optgpu', 'var'),
    optgpu = 0;
end

% % translation
% acc{end+1} = demo_cifar10(optgpu, 8, 1600, 'trans', 2, 0, 2, 0.1, 3, 0.01);
% fname{end+1} = 'translation';

% rotation
% acc{end+1} = demo_cifar10_2nd_layer(optgpu, 24, 8, 'rot', 1, 5, 1, 0.5, 0.1, 0.01);
% fname{end+1} = 'rotation';
acc{end+1} = demo_cifar10_3rd_layer(optgpu, 9, 144, 0.1, 10, 0.01);


% % scale
% acc{end+1} = demo_cifar10(optgpu, 8, 1600, 'scale', 2, 0, 2, 0.1, 3, 0.01);
% fname{end+1} = 'scale';


% print results
% for i = 1:length(acc),
%     fprintf('%s: %g\n', fname{i}, acc{i});
% end


% temp=reshape(hidstate,[24 20449]);
% temp=temp';
% temp=reshape(temp,[143 143 24]);
% test_pool=zeros(72, 72, 24);
% for i = 1:24
% test=temp(1:142,1:142,i);
% for a = 1:2:142
% for b = 1:2:142
% if sum(sum(test(a:a+1,b:b+1)))>0
% test_pool(ceil(a/2),ceil(b/2),i) = 1;
% else
% test_pool(ceil(a/2),ceil(b/2),i) = 0;
% end
% end
% end
% end
% for i = 1:1
% filename = sprintf('hidstates3nd_WB_nowh_(2f40f24f6ws18ws36ws12rP20P10P10Pb01)_%d.mat', i) ;
% load(filename);
% temp=reshape(hidstate,[24 20449]);
% temp=temp';
% 
% temp=reshape(temp,[143 143 24]);
% test_pool=zeros(72, 72, 24);
% for ii = 1:24
% test=temp(1:142,1:142,ii);
% for a = 1:2:142
% for b = 1:2:142
% if sum(sum(test(a:a+1,b:b+1)))>0
% test_pool(ceil(a/2),ceil(b/2),ii) = 1;
% else
% test_pool(ceil(a/2),ceil(b/2),ii) = 0;
% end
% end
% end
% end
% temp2=reshape(test_pool,[72*72,24]);
% display_network(temp2);
% filename2 = sprintf('3rd_layer_state_pool_%d.jpg', i) ;
% %imwrite(temp,filename2,'jpg')
% saveas(gcf,filename2,'jpg');
% end