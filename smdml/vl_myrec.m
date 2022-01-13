function Y = vl_myrec(X, epsilon, dzdy)

% Y = VL_MYREC (X, EPSILON, DZDY)
% ReCov and ReEig layer

Us = cell(length(X),1);
Ss = cell(length(X),1);
Vs = cell(length(X),1);
thres = 1e-6;  % e of the recov layer 
min_v = zeros(1,30);

for ix = 1 : length(X)
    temp = X{ix}; 
    idx1 = temp < 0;
    idx2 = temp > -thres;
    idx = idx1 & idx2;  % indexes of C(i,j) \in (-e, 0]
    temp(idx) = -thres; % set them to -thres
    [Us{ix},Ss{ix},Vs{ix}] = svd(temp);
    min_v(ix) = min(min(diag(Ss{ix})));
end

D = size(Ss{1},2);
Y = cell(length(X),1);

if nargin < 3
    for ix = 1:length(X)
        [max_S, ~] = max_eig(Ss{ix},epsilon); % this equation try to perform relu-like operation
        Y{ix} = Us{ix} * max_S * Us{ix}'; % use the modified eig to re-build the data in this layer
    end
else
    for ix = 1:length(X)
        U = Us{ix}; S = Ss{ix}; V = Vs{ix};

        Dmin = D;
        
        dLdC = double(dzdy{ix}); dLdC = symmetric(dLdC); 
        
        [max_S, max_I] = max_eig(Ss{ix},epsilon); 
        dLdV = 2*dLdC*U*max_S;
        dLdS = (diag(not(max_I)))*U'*dLdC*U; % see eq.18 in SPDNet
        
        
        K = 1./(diag(S)*ones(1,Dmin)-(diag(S)*ones(1,Dmin))'); 
        K(eye(size(K,1))>0)=0; % eq.14 in spdnet
        K(find(isinf(K)==1))=0; 
        
        dzdx = U*(symmetric(K'.*(U'*dLdV))+dDiag(dLdS))*U'; % gradient of the ReEig layer
        
        temp = X{ix};
        idx1 = temp < 0;
        idx2 = temp > -thres; 
        idx = idx1 & idx2; 
        % build the mask M, Eq. 33
        temp(idx) = 0;
        temp(~idx) = 1;
        
        Y{ix} =  temp .* dzdx; % gradient of the ReCov layer
    end
end
