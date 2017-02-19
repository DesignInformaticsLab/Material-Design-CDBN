% -----------------------------------------------------------------------
%   get local transformation matrix - scale
%
%   rs      : receptive field size
%   ws      : window size
%   grid    : spacing
%   ch      : number of channel
%   T       : set of linear transformation matrix,
%             {(ws^2*ch) x (rs^2*ch)}
%
%   e.g., >> T = get_txmat_scale(8, 6, 1, 3)
% -----------------------------------------------------------------------

function T = get_txmat_scale(rs, ws, grid, ch)

if ~exist('ch', 'var'),
    ch = 1;
end
opt = 'bilinear'; % resize option

sc_list = ws:grid:rs; % scale list
nscale = length(sc_list);
T = cell(nscale,1);

n = 0;
for i = 1:nscale,
    in_ws = sc_list(i);
    n = n+1;
    rbuf = floor((rs-in_ws)/2);
    cbuf = floor((rs-in_ws)/2);
    T{n} = get_txmat_scale_sub(in_ws, rs, ws, opt, rbuf, cbuf);
end

% for channel > 1
for i = 1:length(T),
    tmp = zeros(size(T{i},1)*ch,size(T{i},2)*ch);
    for j = 1:ch,
        tmp((j-1)*ws^2+1:j*ws^2,(j-1)*rs^2+1:j*rs^2) = T{i};
    end
    T{i} = sparse(tmp);
end

return;


% -----------------------------------------------------------------------
%   get scale transformation matrix
%   works only with squared input and output
%
%   ws      : actual receptive field window size
%   rs      : receptive field size, rs => ws
%   fs      : filter size, fs <= ws
% -----------------------------------------------------------------------

function T = get_txmat_scale_sub(ws, rs, fs, opt, rbuf, cbuf)

if ~exist('opt', 'var'),
    opt = 'bilinear';
end
if ~exist('rbuf', 'var'),
    rbuf = floor((rs-ws)/2);
end
if ~exist('cbuf', 'var'),
    cbuf = floor((rs-ws)/2);
end

scale = [fs/ws fs/ws];
[weights, indices] = get_resize_weights([ws, ws], [fs, fs], scale, opt);

T = zeros(fs^2, rs^2);
for i = 1:fs,
    for j = 1:fs,
        colidx = indices{1}(i,:);
        rowidx = indices{2}(j,:);
        colw = weights{1}(i,:);
        roww = weights{2}(j,:);
        for s1 = 1:length(colidx),
            for s2 = 1:length(rowidx),
                id = colidx(s1)+cbuf + (rowidx(s2) - 1+rbuf)*rs;
                weight = colw(s1)*roww(s2);
                T(i+(j-1)*fs,id) = T(i+(j-1)*fs,id) + weight;
            end
        end
    end
end
T = sparse(T);

return;