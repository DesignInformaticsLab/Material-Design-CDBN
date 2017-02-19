addpath('utils','function_code','alloy_mat','1st_code')
acc = {};
fname = {};
optgpu=1;
if ~exist('optgpu', 'var'),
    optgpu = 0;
end
%% 1st layer training
tic
fprintf('1st layer training with rotation\n');
demo_cifar10(optgpu, 6, 2, 'rot', 1, 12, 1, 0.1, 20, 0.01);
toc
% hidden layer prepare
addpath('1st_hidstate')
prompt = 'REMIND:\n please change the file name input in "recon_WB.m"(line 2) to the above file name ? \n Y/N [Y]: ';
str = input(prompt,'s');

run recon_WB
prompt = 'REMIND:\n please make sure "fname"(line 3) in "1st_hidstate//combine_1st" matches the file name in 1st_hidstate\n Y/N [Y]: ';
str = input(prompt,'s');
run combine_1st %pooling + combining
clear

fprintf('2nd layer RBM is ready...\n');
%% 2nd layer training
prompt = 'REMIND:\n please change the file name in "load_patches_2nd_layer.m"(line 18) to the above file name ? \n Y/N [Y]: ';
str = input(prompt,'s');
tic
acc = {};fname = {};optgpu=1;
addpath('2nd_code')
acc{end+1} = WB_2nd_layer(optgpu, 9, 40, 0.1, 20, 0.01);
fname{end+1} = 'rotation';
toc
% hidden layer prepare
addpath('2nd_hidstate','2nd_code')
prompt = 'REMIND:\n please change the file name in "second_recon_.m"(line 2,line 38) to the above file name and the pooled hidstate file name? \n Y/N [Y]: ';
str = input(prompt,'s');
run second_recon_
prompt = 'REMIND:\n please make sure "fname"(line 3) in "2nd_hidstate//combine_2nd" matches the file name in 1st_hidstate\n Y/N [Y]: ';
str = input(prompt,'s');
clear
run combine_2nd
fprintf('3rd layer RBM is ready...\n');
%% 3rd layer training
clear
prompt = 'REMIND:\n please change the file name in "load_patches_3rd_layer.m"(line 18) to the above file name ? \n Y/N [Y]: ';
str = input(prompt,'s');
tic
acc = {};fname = {};optgpu=1;
addpath('3rd_code','3rd_hidstate')
acc{end+1} = WB_3rd_layer(optgpu, 9, 288, 0.1, 10, 0.01);
toc
prompt = 'REMIND:\n please change the file name in "third_recon_.m"(line 2,line 30) to the above file name and the pooled hidstate file name? \n Y/N [Y]: ';
str = input(prompt,'s');
run third_recon_
prompt = 'REMIND:\n Please do the same thing as in 2nd layer? \n Y/N [Y]: ';
str = input(prompt,'s');
run combine_3rd
prompt = 'REMIND:\n Please do the same thing as in 2nd layer? \n Y/N [Y]: ';
str = input(prompt,'s');
fprintf('4th layer RBM is ready...\n');
%% 4th layer training
clear
prompt = 'REMIND:\n Please do the same thing as in former layer? \n Y/N [Y]: ';
str = input(prompt,'s');
tic
acc = {};fname = {};optgpu=1;
addpath('4th_code','4th_hidstate')
acc{end+1} = WB_4th_layer(optgpu, 36, 1000, 0.1, 10, 0.01);
toc
prompt = 'REMIND:\n Please do the same thing as in former layer? \n Y/N [Y]: ';
str = input(prompt,'s');
run rbm4th_recon_
prompt = 'REMIND:\n Please do the same thing as in former layer? \n Y/N [Y]: ';
str = input(prompt,'s');
run combine_4th
prompt = 'REMIND:\n Please do the same thing as in former layer? \n Y/N [Y]: ';
str = input(prompt,'s');
fprintf('5th layer RBM is ready...\n');
%% 5th layer training
clear
prompt = 'REMIND:\n Please do the same thing as in former layer? \n Y/N [Y]: ';
str = input(prompt,'s');
tic
addpath('5th_code','5th_hidstate')
acc = {};fname = {};optgpu=1;
acc{end+1} = WB_5th_layer(optgpu, 1, 30, 0, 0, 0.01);
toc
prompt = 'REMIND:\n Please do the same thing as in former layer? \n Y/N [Y]: ';
str = input(prompt,'s');
run rbm5th_recon_
fprintf('Training Part is Done...\n');