% -----------------------------------------------------------------------
%   Get transformation matrix
%
%   tx_type     : transformation type
%                 'trans', 'scale', 'rot'
%   rs          : receptive field size
%   ws          : window size (ws = rs for rot, rs > ws for trans, scale)
%   grid        : spacing for translation and scale transformation
%   numrot      : number of rotation
% -----------------------------------------------------------------------

function Tlist = get_txmat(tx_type, rs, ws, grid, numrot, numch)

if strcmp(tx_type, 'trans'),
    % translation
    Tlist = get_txmat_trans(rs, ws, grid, numch);
elseif strcmp(tx_type, 'scale'),
    % scale
    Tlist = get_txmat_scale(rs, ws, grid, numch);
elseif strcmp(tx_type, 'rot'),
    % rotation
    ang_list = get_angles(numrot);
    Tlist = get_txmat_rot(rs, numch, ang_list);
end

return;


% -----------------------------------------------------------------------
%   Get list of rotation angles
% -----------------------------------------------------------------------

function ang_list = get_angles(numrot)

offset = ceil(numrot/2);
ang_list = (1-offset:numrot-offset)*pi/12;

return