function Y = vl_mylog(X, dzdy)

% Y = VL_MYLOG(X, DZDY)
% LogEig layer

Us = cell(length(X),1);
Ss = cell(length(X),1);
Vs = cell(length(X),1);

for ix = 1 : length(X)
    [Us{ix},Ss{ix},Vs{ix}] = svd(X{ix});
end


D = size(Ss{1},2);
Y = cell(length(X),1);

if nargin < 2  % forward computing
    for ix = 1:length(X)
        Y{ix} = Us{ix}*diag(log(diag(Ss{ix})))*Us{ix}'; % reconstruction the data via the log-eig
    end
else % used for backpropgation 1) X = UZU'; 2) log(X) = Ulog(Z)U'
    for ix = 1:length(X)
        U = Us{ix}; S = Ss{ix}; V = Vs{ix};
        diagS = diag(S);
        ind = diagS >(D*eps(max(diagS))); % judge the eigenvalues are positive or not 
        Dmin = (min(find(ind,1,'last'),D));
        
        S = S(:,ind); U = U(:,ind);
        
        dLdC = double(reshape(dzdy(:,ix),[D D])); 
        dLdC = symmetric(dLdC); % (A + A') / 2       
        
        dLdV = 2*dLdC*U*diagLog(S,0); % Eq.47 in DeepO2P
        dLdS = diagInv(S)*(U'*dLdC*U); % Eq.47 in DeepO2P
        
        if sum(ind) == 1 % diag behaves badly when there is only 1d
            K = 1./(S(1)*ones(1,Dmin)-(S(1)*ones(1,Dmin))'); 
            K(eye(size(K,1))>0)=0; 
        else
            K = 1./(diag(S)*ones(1,Dmin)-(diag(S)*ones(1,Dmin))'); % first line of Eq.37 in DeepO2P
            K(eye(size(K,1))>0)=0; % the second line of Eq.37 in DeepO2P
            K(find(isinf(K)==1))=0; % 
        end                                                                
        
        if all(diagS==1)
            dzdx = zeros(D,D);
        else
            dzdx = U*(symmetric(K'.*(U'*dLdV))+dDiag(dLdS))*U'; % Eq.39 in DeepO2P
        end
        
        Y{ix} =  dzdx; % warning('no normalization');
   
    end
end
