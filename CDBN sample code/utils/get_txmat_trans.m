% -----------------------------------------------------------------------
%   get local transformation matrix - translation
%
%   rs      : receptive field size
%   ws      : window size
%   grid    : spacing
%   ch      : number of channel
%   T       : set of linear transformation matrix,
%             {(ws^2*ch) x (rs^2*ch)}
%
%   e.g., >> T = get_txmat_trans(8, 6, 1, 3)
% -----------------------------------------------------------------------

function T = get_txmat_trans(rs, ws, grid, ch)

if ~exist('ch', 'var'),
    ch = 1;
end
tx_list = 0:grid:abs(rs - ws);

T = cell(length(tx_list)^2, 1);

k = 0;
for i = 1:length(tx_list),
    for j = 1:length(tx_list),
        k = k + 1;
        nx = tx_list(i);
        ny = tx_list(j);
        T{k} = get_txmat_trans_sub(rs, ws, nx, ny)';
    end
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
%   get translation transformation matrix
%   works only with squared input and output
%
%   rs      : receptive field size
%   ws      : window size
%   nx      : x-offset (>= 0)
%   ny      : y-offset (>= 0)
% -----------------------------------------------------------------------

function T = get_txmat_trans_sub(rs, ws, nx, ny)

T = zeros(rs^2,ws^2);
for i = 1:ws,
    for j = 1:ws,
        T((ny+j-1)*rs+(nx+i),(j-1)*ws+i) = 1;
    end
end
T = sparse(T);

return;