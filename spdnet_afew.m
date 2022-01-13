function [net, info] = spdnet_afew(varargin)

% set up the path
confPath; % upload the path of some toolkits to the workspace

% parameter setting
opts.dataDir = fullfile('./data/afew') ;
opts.imdbPathtrain = fullfile(opts.dataDir, 'SPD_info.mat'); 
opts.batchSize = 30; 
opts.test.batchSize = 1;
opts.numEpochs = 1120; 
opts.gpus = [] ;
opts.learningRate = 0.01 * ones(1,opts.numEpochs); 
opts.weightDecay = 0.0005 ; 
opts.continue = 1;

% smdml initialization and the construction of the basic architecture
net = spdnet_init_afew_deep_v1() ; % 

% loading metadata 
load(opts.imdbPathtrain) ;

% training and testing
[net, info] = spdnet_train_afew(net, spd_train, opts);