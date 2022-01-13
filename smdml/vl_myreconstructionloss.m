function Y = vl_myreconstructionloss(X, X_ori, dzdy)
  
  % this function is designed to implement the reconstruction error term
  % Date: 
  % Authors:
  % Coryright@
  
  for m = 1 : length(X)
      dzdy{m} = single(zeros(size(X{1},1),size(X{1},1))); 
  end
  
  dzdy_l3 = single(1);
  rho = 0.1; % needs to be adjusted
  dist_sum = zeros(1,length(X)); 
  Y = cell(length(X), 1); % save obj or dev
  dev_term = cell(1, length(X)); % save each pair' derivation 
  
  for i = 1 : length(X)
      temp = X{i} - X_ori{i};
      dev_term{i} = 2 * temp;
      dist_sum(i) = norm(temp,'fro') * norm(temp,'fro'); 
  end
  
  if nargin < 3
      Y = rho * (sum(dist_sum) / length(X)); % the obj of this loss function
  else
      for j = 1 : length(X)
          dev_l3 = bsxfun(@times, dev_term{j}, bsxfun(@times, ones(size(X{1},1)), dzdy_l3));
          Y{j} = rho * dev_l3 + dzdy{j}; % gradient of the RT
      end
  end
  
end

