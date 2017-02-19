function [patch] = noload_patches(ws)
prepare_cifar10;

% Load CIFAR training data
fprintf('Loading training data...\n');
f1=load([CIFAR_DIR '/Scaled_Gray.mat']);
xtr = double([f1.Scaled_Gray;]);
clear f1;
L=sqrt(size(xtr,2));H=sqrt(size(xtr,2));
fname = sprintf('Noload_Patches_Scaled_Gray_alloy_ws_%d',ws);
xb = zeros(size(xtr,1)*(L-ws+1)*(H-ws+1),ws^2);
try
    load(sprintf('patch/%s.mat', fname));
catch
    for i = 1 : size(xtr,1)
        if (mod(i,3) == 0),
            fprintf('Extracting image: %d', i);
        end
        temp = reshape(xtr(i,:),[sqrt(size(xtr,2)),sqrt(size(xtr,2))]);
        for m = 1 : (L-ws+1)
            for n = 1 : (H-ws+1)               
                 temp2 = temp(m:ws+m-1,n:ws+n-1);
                 xb(n+(L-ws+1)*(m-1)+(L-ws+1)*(H-ws+1)*(i-1),:) = reshape(temp2,[1,ws*ws]);
            end
        end
    end
end
patch = xb;
save(sprintf('patch/%s.mat', fname))
return