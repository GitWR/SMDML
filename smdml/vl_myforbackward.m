function [res] = vl_myforbackward(net, x, dzdy, res, epoch, count1, varargin)

% implementation of the forward and backward pass of the proposed SMDML

opts.res = [] ;
opts.conserveMemory = false ;
opts.sync = false ;
opts.disableDropout = false ;
opts.freezeDropout = false ;
opts.accumulate = false ;
opts.cudnn = true ;
opts.skipForward = false;
opts.backPropDepth = +inf ;
opts.epsilon = 1e-5; % this parameter: $\xi$, is worked in the rec Layer

n = numel(net.layers) ; % count the number of layers

if (nargin <= 2) || isempty(dzdy)
  doder = false ;
else
  doder = true ; % this variable is used to control when stepping into backpropagation
end

if opts.cudnn
  cudnn = {'CuDNN'} ;
else
  cudnn = {'NoCuDNN'} ;
end

gpuMode = isa(x, 'gpuArray') ;

if nargin <= 3 || isempty(res)
  res = struct(...
    'x', cell(1,n+1), ...
    'dzdx', cell(1,n+1), ... % this is necessary for computing the gradients in the layers below and updating their parameters  
    'dzdw', cell(1,n+1), ... % this is required for updating W
    'aux', cell(1,n+1), ...
    'time', num2cell(zeros(1,n+1)), ...
    'backwardTime', num2cell(zeros(1,n+1))) ;
end
if ~opts.skipForward
  res(1).x = x ;
end


% -------------------------------------------------------------------------
%                                                              Forward pass
% -------------------------------------------------------------------------

for i=1:n
  if opts.skipForward
      break; 
  end
  l = net.layers{i} ; % 
  res(i).time = tic ; % 
  switch l.type
    case 'bfc'
      res(i+1).x = vl_mybfc(res(i).x, l.weight, i, res) ; % the output data of each layer is stored in the x part
    case 'fc'
      res(i+1).x = vl_myfc(res(i).x, l.weight) ;
    case 'rec'
      res(i+1).x = vl_myrec(res(i).x, opts.epsilon) ;
    case 'rec_relu'
      res(i+1).x = vl_myrec_relu(res(i).x, opts.epsilon) ;
    case 'marginloss'
      if doder
        [res(i+1).obj, WW, BB] = vl_mymarginloss(res(i).x, l.class, epoch, count1, doder) ;  % this is the metric learning regularizer
        res(i+1).ww = WW;
        res(i+1).bb = BB;
      else
        res(i+1).obj = 0;
      end
      res(i+1).x = res(i).x;
    case 'reconstructionloss'
      res(i+1).obj = vl_myreconstructionloss(res(i).x, res(1).x); % reconstruction error term
      res(i+1).x = res(7).x;
    case 'log'
      res(i+1).x = vl_mylog(res(i).x) ;
    case 'softmaxloss'
      res(i+1).x = vl_mysoftmaxloss(res(i).x, l.class) ;
    case 'custom'
          res(i+1) = l.forward(l, res(i), res(i+1)) ;
    otherwise
      error('Unknown layer type %s', l.type) ;
  end
  % optionally forget intermediate results
  forget = opts.conserveMemory ;
  forget = forget & (~doder || strcmp(l.type, 'relu')) ;
  forget = forget & ~(strcmp(l.type, 'loss') || strcmp(l.type, 'softmaxloss')) ;
  forget = forget & (~isfield(l, 'rememberOutput') || ~l.rememberOutput) ;
  if forget
    res(i).x = [] ;
  end
  if gpuMode & opts.sync
    % This should make things slower, but on MATLAB 2014a it is necessary
    % for any decent performance.
    wait(gpuDevice) ;
  end
  res(i).time = toc(res(i).time) ;
end

% -------------------------------------------------------------------------
%                                                             Backward pass
% -------------------------------------------------------------------------

if doder
  res(n+1).dzdx = dzdy ; % 
  for i = n:-1:max(1, n-opts.backPropDepth+1) %
    l = net.layers{i} ;
    res(i).backwardTime = tic ;
    switch l.type
      case 'bfc'
        [res(i).dzdx, res(i).dzdw] = ... % 
             vl_mybfc(res(i).x, l.weight, i, res, res(i+1).dzdx) ; % 
                                                           
      case 'fc'
        [res(i).dzdx, res(i).dzdw]  = ...
              vl_myfc(res(i).x, l.weight, res(i+1).dzdx) ; 
      case 'rec'
        res(i).dzdx = vl_myrec(res(i).x, opts.epsilon, res(i+1).dzdx) ;
      case 'rec_relu'
        res(i).dzdx = vl_myrec_relu(res(i).x, opts.epsilon, res(i+1).dzdx) ;
      case 'marginloss'
        alt = doder;
        alt = alt-1;
        [res(i).dzdx, ~, ~] = vl_mymarginloss(res(i).x, l.class, epoch, count1, alt, res(i+1).ww, res(i+1).bb, res(i+1).dzdx, res(i+7).dzdx) ;
      case 'reconstructionloss'
        res(i).dzdx = vl_myreconstructionloss(res(i).x, res(1).x, res(i+1).dzdx) ;
      case 'log'
        res(i).dzdx = vl_mylog(res(i).x, res(i+1).dzdx) ;
      case 'softmaxloss'
        res(i).dzdx = vl_mysoftmaxloss(res(i).x, l.class, res(i+1).dzdx) ;
      case 'custom'
        res(i) = l.backward(l, res(i), res(i+1));
    end
    if opts.conserveMemory
      res(i+1).dzdx = [] ;
    end
    if gpuMode & opts.sync
      wait(gpuDevice) ;
    end
    res(i).backwardTime = toc(res(i).backwardTime) ;
  end
end

