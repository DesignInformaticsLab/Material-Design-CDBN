function [label,center] = litekmeans(X, k, opt_verbose, MAX_ITERS)
if ~exist('./litekmeans_results.mat','file')
    if ~exist('opt_verbose', 'var')
        opt_verbose = false;
    end

    if ~exist('MAX_ITERS', 'var')
        MAX_ITERS = 50;
    end

    X=X';
    n = size(X,2);
    last = 0;
    label = ceil(k*rand(1,n));  % random initialization
    itr=0;
    % MAX_ITERS=50;
    while any(label ~= last)
        itr = itr+1;
        if opt_verbose
            fprintf(1, '%d(%d)..', itr, sum(label ~= last));
        end

        E = sparse(1:n,label,1,n,k,n);  % transform label into indicator matrix
        center = X*(E*spdiags(1./sum(E,1)',0,k,k));    % compute center of each cluster
        last = label;
        [val,label] = max(bsxfun(@minus,center'*X,0.5*sum(center.^2,1)')); % assign samples to the nearest centers
        if (itr >= MAX_ITERS) break; end;
    end
    center=center';
    save('litekmeans_results.mat','center','label'); %TODO: make the file name more general
else
    load('litekmeans_results.mat','center','label');
end
if opt_verbose
    fprintf(1,'\n');
end

