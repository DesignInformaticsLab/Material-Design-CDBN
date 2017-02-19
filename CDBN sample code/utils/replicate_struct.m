% ===================================
% replicate struct variables
% ===================================


function B = replicate_struct(A, val)

if ~exist('val', 'var'),
    val = 1;
end

B = struct;
fname = fieldnames(A);

for i = 1:length(fname),
    a = getfield(A, fname{i});
    B = setfield(B, fname{i}, val*a);
end

return;