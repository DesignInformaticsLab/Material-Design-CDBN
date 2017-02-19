% ======================================
% convert struct variables from 
% gpu (jacket) to cpu
% ======================================


function A = gpu2cpu_struct(A)

fname = fieldnames(A);
for i = 1:length(fname),
    a = getfield(A, fname{i});
    A = setfield(A, fname{i}, gather(a));
end

return;