% -----------------------------------------------------------------------
%   get local transformation matrix - rotation
%
%   ws      : window size
%   ch      : number of channel
%   ang_list: list of rotation angles
%   T       : set of linear transformation matrix,
%             {(ws^2*ch) x (ws^2*ch)}
%
%   e.g., >> T = get_txmat_rot(6, 3)
% -----------------------------------------------------------------------

function T = get_txmat_rot(ws, ch, ang_list)

if ~exist('ch', 'var'),
    ch = 1;
end
if ~exist('ang_list', 'var'),
    ang_list = (-4:2:4)*pi/24;
end

T = cell(length(ang_list), 1);

% compute transformation matrix
for ang_id = 1:length(ang_list),
    ang = ang_list(ang_id);
    T{ang_id} = get_txmat_rot_sub(ws,ang);
end

% for channel > 1
for i = 1:length(T),
    tmp = zeros(ws^2*ch,ws^2*ch);
    for j = 1:ch,
        tmp((j-1)*ws^2+1:j*ws^2,(j-1)*ws^2+1:j*ws^2) = T{i};
    end
    T{i} = sparse(tmp);
end

return;


% -----------------------------------------------------------------------
%   get rotation transformation matrix
%   works only with squared input and output
%
%   ws      : window size
%   ang     : angle (radian, e.g., 2*pi*(ang_id-1)/(# transformation)
% -----------------------------------------------------------------------

function T = get_txmat_rot_sub(ws, ang)

cX = ws/2;
cY = ws/2;
T = zeros(ws^2,ws^2);
for i= 1:ws,
    relY = i-cY-0.5;		% get center y coordinate
    for j= 1:ws,
        relX = j-cX-0.5;	% get center x coordinate
        id = i + (j-1)*ws;
        % do rotation transformation
        xPrime = relX*cos(ang) + relY*sin(ang);
        yPrime = -1 * relX*sin(ang) + relY*cos(ang);
        % re-center pixel
        xPrime = xPrime + cX + 0.5;
        yPrime = yPrime + cY + 0.5;
        
        % four coordinate
        q12x = floor(xPrime);
        q12y = floor(yPrime);
        q12x = max(1, q12x);
        q12y = max(1, q12y);
        q12x = min(ws, q12x);
        q12y = min(ws, q12y);
        q22x = ceil(xPrime);
        q22y = q12y;
        q22x = min(ws, q22x);
        q22x = max(1, q22x);
        q11x = q12x;
        q11y = ceil(yPrime);
        q11y = min(ws, q11y);
        q11y = max(1, q11y);
        q21x = q22x;
        q21y = q11y;
        
        % find coordinate
        q11 = (q11x-1)*ws + q11y;
        q12 = (q12x-1)*ws + q12y;
        q21 = (q21x-1)*ws + q21y;
        q22 = (q22x-1)*ws + q22y;
        
        % compute ratio
        if ( q21x == q11x ) % special case to avoid divide by zero
            factor1 = 1;    % They're at the same X coordinate, so just force the calculatione to one point
            factor2 = 0;
        else
            factor1 = (q21x - xPrime)/(q21x - q11x);
            factor2 = (xPrime - q11x)/(q21x - q11x);
        end
        
        if (q12y == q11y) % special case to avoid divide by zero
            factor3 = 1;
            factor4 = 0;
        else
            factor3 = (q12y - yPrime)/(q12y - q11y);
            factor4 = (yPrime - q11y)/(q12y - q11y);
        end
        if q21x ~= q11x && q12y ~= q11y,
            T(id,q11) = factor1 * factor3;
            T(id,q21) = factor2 * factor3;
            T(id,q12) = factor1 * factor4;
            T(id,q22) = factor2 * factor4;
        elseif q21x == q11x && q12y ~= q11y,
            T(id,q11) = factor1 * factor3;
            T(id,q12) = factor1 * factor4;
        elseif q21x ~= q11x && q12y == q11y,
            T(id,q11) = factor1 * factor3;
            T(id,q21) = factor2 * factor3;
        else
            T(id,q11) = factor1 * factor3;
        end
    end
end
T = sparse(T);

return;
