function net = spdnet_init_afew_deep_v1(varargin)
% this function contains the initialization of the filter banks and the
% construction of the basic architecture of the proposed SMDML

rng('default');
rng(0) ;

opts.layernum = 6; % denotes the number of the BiMap layers

Winit = cell(opts.layernum,1);
opts.datadim = [63,53,43,33,43,53,63]; % the dimensionality of each bimap layer, which also indicates the kernel size of each layer

for iw = 1 : opts.layernum 
    if iw < 4
       A = rand(opts.datadim(iw));
       [U1, ~, ~] = svd(A * A');
       Winit{iw} = U1(:,1:opts.datadim(iw+1)); % the initialized filters are all satisfy column orthogonality
    else
       A = rand(opts.datadim(iw+1));
       [U1, ~, ~] = svd(A * A');
       temp = U1(:,1:opts.datadim(iw));
       Winit{iw} = temp';
    end
end

f = 1/100 ;
classNum = 45; % number of categories
fdim = size(Winit{iw-3},2) * size(Winit{iw-3},2);
theta = f * randn(fdim, classNum, 'single');
Winit{end+1} = theta; % the FC layer's weight

net.layers = {} ; % use to construct each layer of the proposed SMDML
net.layers{end+1} = struct('type', 'bfc',...
                          'weight', Winit{1}) ;
net.layers{end+1} = struct('type', 'rec') ; 
net.layers{end+1} = struct('type', 'bfc',...
                          'weight', Winit{2}) ;
net.layers{end+1} = struct('type', 'rec') ;
net.layers{end+1} = struct('type', 'bfc',...
                          'weight', Winit{3}) ;
net.layers{end+1} = struct('type', 'marginloss') ;
net.layers{end+1} = struct('type', 'bfc',...
                          'weight', Winit{4}) ;
net.layers{end+1} = struct('type', 'rec') ;
net.layers{end+1} = struct('type', 'bfc',...
                          'weight', Winit{5}) ;
net.layers{end+1} = struct('type', 'rec') ; 
net.layers{end+1} = struct('type', 'bfc',...
                          'weight', Winit{6}) ;
net.layers{end+1} = struct('type', 'reconstructionloss') ;
net.layers{end+1} = struct('type', 'log') ;
net.layers{end+1} = struct('type', 'fc', ...
                           'weight', Winit{end}) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;

