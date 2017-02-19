% -------------------------------------------------------------------------
% draw sample from multinomial distribution x
%
%   x   : numlab x batchsize (distribution)
%   y   : numlab x batchsize (sample)
% -------------------------------------------------------------------------

function [y, x] = sample_multinomial(x, optgpu)

if ~exist('optgpu','var'),
    optgpu = 0;
end

if optgpu,
    x = gather(x);
end

x = double(x)';
x = bsxfun(@rdivide, x, sum(x, 2));

cumx = cumsum(x, 2);
unifrnd = rand(size(x, 1), 1);
temp = bsxfun(@gt, cumx, unifrnd);
yidx = diff(temp, 1, 2);
y = zeros(size(x));
y(:,1) = 1-sum(yidx,2);
y(:,2:end) = yidx;

x = x';
y = y';

if optgpu,
    y = gpuArray(y);
    x = gpuArray(x);
end

return;
