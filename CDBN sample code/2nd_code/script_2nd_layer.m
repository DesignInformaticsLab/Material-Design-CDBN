acc = {};
fname = {};
optgpu = 1;
if ~exist('optgpu', 'var'),
    optgpu = 0;
end


acc{end+1} = demo_cifar10_2nd_layer(optgpu, 18, 480, 0.1, 10, 0.01);
