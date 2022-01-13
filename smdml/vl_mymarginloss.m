function [Y, parW, parB] = vl_mymarginloss(X, c, epoch, count1, doder, parW, parB, dzdy_recon, dzdx_log)

% metric learning term

tau1 = -1.0; % a throshold to control the inter-class manifold margin 
tau2 = 1.0; % a throshold to control the intra-class manifold margin

new_count = count1;
dzdy_l2 = single(1);

eta = 0.01; % trade-off parameter of the MLT
eta = 0.8^floor(epoch / 200) * eta;

Nw = zeros(1, length(X));
Nb = zeros(1, length(X));
Sw = zeros(1, length(X)); % intra-class scatter
Sb = zeros(1, length(X)); % inter-class scatter

temp_dev_Sw = zeros(size(X{1},1),size(X{1},1),length(X));
temp_dev_Sb = zeros(size(X{1},1),size(X{1},1),length(X));

use_parfor = cell(1,length(X));
for i = 1 : size(use_parfor,2)
    use_parfor{i} = X;
end

if doder
  parfor j = 1 : length(X)
    K1 = 4; % the number of intra-manifold neighboring points
    num_eachclass = find(c==c(j));
    temp_X = use_parfor{j};
    Xi = temp_X{j};
    Sw_temp = zeros(1,length(num_eachclass));
    temp_dev_Sw_store = zeros(size(X{1},1),size(X{1},1),length(num_eachclass));
    for k = 1 : length(num_eachclass)
        Nw(j) = Nw(j) + 1;
        Xj = temp_X{num_eachclass(k)};
        [uxi,vxi,txi] = svd(Xi);
        [uxj,vxj,txj] = svd(Xj);
        logm_xi = uxi*diag(log(diag(vxi)))*txi';
        logm_xj = uxj*diag(log(diag(vxj)))*txj';
        temp = logm_xi - logm_xj;
        temp_dev_Sw_store(:,:,k) = Xi\temp;
        Sw_temp(k) = norm(temp, 'fro') * norm(temp, 'fro'); 
    end
    [~,idx] = sort(Sw_temp);
    if (length(num_eachclass) < K1)
        K1 = length(num_eachclass);
    end
    dist_temp = Sw_temp(idx(:,1:K1)); % get the first K1 smallest distances
    Sw(j) = sum(dist_temp); % this is the scatter matrix composed by K1 nearest samples
    temp_dev_Sw(:,:,j) = sum(temp_dev_Sw_store(:,:,idx(:,1:K1)),3); 
    if K1 == 1
        Nw(j) = K1 +1;
    else 
        Nw(j) = K1;
    end
  end

  parfor j = 1:length(X)
    K2 = 3; % the number of the inter-manifold neighboring points
    num_difclass=find(c~=c(j)); %
    temp_X = use_parfor{j};
    Xi = temp_X{j};
    Sb_temp = zeros(1,length(num_difclass));
    temp_dev_Sb_store = zeros(size(X{1},1),size(X{1},1),length(num_difclass));
    for k = 1:length(num_difclass)
        Xj = temp_X{num_difclass(k)};
        Nb(j) = Nb(j) + 1;
        [uxi,vxi,txi] = svd(Xi);
        [uxj,vxj,txj] = svd(Xj);
        logm_xi = uxi*diag(log(diag(vxi)))*txi';
        logm_xj = uxj*diag(log(diag(vxj)))*txj';
        temp = logm_xi - logm_xj;
        temp_dev_Sb_store(:,:,k) = Xi\temp;
        Sb_temp(k) = norm(temp, 'fro') * norm(temp, 'fro');
    end
    [~,idx] = sort(Sb_temp);
    if (length(num_difclass) < K2)
        K2 = length(num_difclass);
    end
    dist_temp = Sb_temp(idx(:,1:K2)); % get the first K2 smallest distances
    Sb(j) = sum(dist_temp); % this is the scatter matrix composed by K2 nearest samples
    temp_dev_Sb(:,:,j) = sum(temp_dev_Sb_store(:,:,idx(:,1:K2)),3); 
    Nb(j) = K2;
  end
  parW.Sw = Sw;
  parW.Nw = Nw;
  parW.temp_dev_Sw = temp_dev_Sw;
  parB.Sb = Sb;
  parB.Nb = Nb;
  parB.temp_dev_Sb = temp_dev_Sb;
else
  Sw = parW.Sw;
  Nw = parW.Nw;
  temp_dev_Sw = parW.temp_dev_Sw;
  Sb = parB.Sb;
  Nb = parB.Nb;
  temp_dev_Sb = parB.temp_dev_Sb;
end

beta = 0.2; 

temp_scatter = zeros(1,length(X));
Y = cell(length(X), 1);
Y_sum = 0;

for m = 1 : length(X)
    
    Sw_each = Sw(m) / (Nw(m)-1);
    Sb_each = Sb(m) / Nb(m);  
    d_inter = Sw_each - Sb_each;
    d_intra = Sw_each;
    % temp_scatter(m) = d_inter + alpha * d_intra;
    % Y_sum = Y_sum + log(1 + exp(temp_scatter(m)));
    if (d_inter <= tau1 && d_intra <= tau2)  
        temp_scatter(m) = tau1 + beta * tau2;
        if temp_scatter(m)>= 700
            temp_scatter(m) = temp_scatter(m)/1.5;
        end
        Y_sum = Y_sum + log(1 + exp(temp_scatter(m)));
        if nargin < 6
            Y = eta * Y_sum;
        else
            Y{m} = 0 + dzdy_recon{m} + dzdx_log{m}; % 0 --> (3).1) of Section III-E
        end
    elseif d_inter <= tau1 && d_intra > tau2
        temp_scatter(m) = tau1 + beta * d_intra;
        if temp_scatter(m)>= 700
            temp_scatter(m) = temp_scatter(m)/1.5;
        end
        Y_sum = Y_sum + log(1 + exp(temp_scatter(m)));
        if nargin < 6
            Y = eta * Y_sum;
        else
            dev_part1 = 1 / (1 + exp(temp_scatter(m)));
            dev_part2 = exp(temp_scatter(m)); 
            temp_dev_Sw_each = temp_dev_Sw(:,:,m) / (Nw(m) - 1);
            dev_part3 = 2 * beta * temp_dev_Sw_each;
            dev_l2 = bsxfun(@times, (dev_part1 * dev_part2 * dev_part3), bsxfun(@times, ones(size(X{1},1)), dzdy_l2));
            Y{m} = eta * dev_l2 + dzdy_recon{m} + dzdx_log{m};  % dev_l2 --> (3).2) of Section III-E
        end
    elseif d_inter > tau1 && d_intra <= tau2
        temp_scatter(m) = d_inter + beta * tau2;
        if temp_scatter(m)>= 700
            temp_scatter(m) = temp_scatter(m)/1.5;
        end
        Y_sum = Y_sum + log(1 + exp(temp_scatter(m)));
        if nargin < 6
            Y = eta * Y_sum;
        else
            dev_part1 = 1 / (1 + exp(temp_scatter(m)));
            dev_part2 = exp(temp_scatter(m)); 
            temp_dev_Sw_each = temp_dev_Sw(:,:,m) / (Nw(m) - 1);
            temp_dev_Sb_each = temp_dev_Sb(:,:,m) / Nb(m);
            dev_part3 = 2 * temp_dev_Sw_each - 2 * temp_dev_Sb_each;
            dev_l2 = bsxfun(@times, (dev_part1 * dev_part2 * dev_part3), bsxfun(@times, ones(size(X{1},1)), dzdy_l2));
            Y{m} = eta * dev_l2 + dzdy_recon{m} + dzdx_log{m};  % dev_l2 --> (3).3) of Section III-E
        end
    else
        temp_scatter(m) = d_inter + beta * d_intra;
        if temp_scatter(m)>= 700
            temp_scatter(m) = temp_scatter(m)/1.5;
        end
        Y_sum = Y_sum + log(1 + exp(temp_scatter(m)));
        if nargin < 6
            Y = eta * Y_sum;
        else
            dev_part1 = 1 / (1 + exp(temp_scatter(m)));
            dev_part2 = exp(temp_scatter(m)); 
            temp_dev_Sw_each = temp_dev_Sw(:,:,m) / (Nw(m) - 1);
            temp_dev_Sb_each = temp_dev_Sb(:,:,m) / Nb(m);
            dev_part3 = 2 * temp_dev_Sw_each - 2 * temp_dev_Sb_each + 2 * beta * temp_dev_Sw_each;
            dev_l2 = bsxfun(@times, (dev_part1 * dev_part2 * dev_part3), bsxfun(@times, ones(size(X{1},1)), dzdy_l2));
            Y{m} = eta * dev_l2 + dzdy_recon{m} + dzdx_log{m};  % dev_l2 --> (3).4) of Section III-E
        end
    end
    
end



