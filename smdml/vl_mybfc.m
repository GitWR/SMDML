function [Y, Y_w] = vl_mybfc(X, W, count, res, dzdy)

% BiMap layer  

Y = cell(length(X),1); % used to store the result data of bimap layer 

% skip conneciton, but our SMDML does not do this
skip_data = res(count).x;

% bimap computation
for ix = 1 : length(X)
    Y{ix} = W' * X{ix} * W ;
end

% partial derivative computation
if nargin == 5
    [dim_ori, dim_tar] = size(W);
    Y_w = zeros(dim_ori,dim_tar);
    for ix = 1  : length(X)
        if iscell(dzdy)==1
            d_t = dzdy{ix};
        else
            d_t = dzdy; %(:,ix);
            % d_t = reshape(d_t,[dim_tar dim_tar]);
        end
        Y{ix} = W * d_t * W'; % dzdx
        Y_w = Y_w + 2 * X{ix} * W * d_t; % dzdw
    end
end
