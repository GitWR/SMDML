function [net, info] = spdnet_train_afew(net, spd_train, opts)

opts.errorLabels = {'top1e'};
opts.train = find(spd_train.spd.set==1) ; % 1 represents the training samples
opts.val = find(spd_train.spd.set==2) ; % 2 indicates the test samples

for epoch = 1 : opts.numEpochs 
    
     learningRate = opts.learningRate(epoch); 
     
     %% fast-forward to last checkpoint
     modelPath = @(ep) fullfile(opts.dataDir, sprintf('net-epoch-%d.mat', ep));
     modelFigPath = fullfile(opts.dataDir, 'net-train.pdf') ;
     if opts.continue
         if exist(modelPath(epoch),'file')
             if epoch == opts.numEpochs
                 load(modelPath(epoch), 'net', 'info') ;
             end
             continue ;
         end
         if epoch > 1
             fprintf('resuming by loading epoch %d\n', epoch-1) ;
             load(modelPath(epoch-1), 'net', 'info') ;
         end
     end
     
    train = opts.train(randperm(length(opts.train))) ; % data_label; shuffling, to make the training data feed into the net in disorder
    val = opts.val; 
    [net,stats.train] = process_epoch(opts, epoch, spd_train, train, learningRate, net) ; % training stage
    [net,stats.val] = process_epoch(opts, epoch, spd_train, val, 0, net) ; % test stage
        
   %% the following is to dynamicly draw the cost and error curve
   evaluateMode = 0;
     if evaluateMode
         sets = {'train'};
     else
         sets = {'train', 'val'};
     end
     for f = sets
         f = char(f);
         n = numel(eval(f));
         info.(f).objective(epoch) = (stats.(f)(2) + stats.(f)(3) + stats.(f)(4)) / n; % stats.(f)(2) / n;
         info.(f).acc(:,epoch) = stats.(f)(5:end) / n; % stats.(f)(3:end) / n;
     end
     if ~evaluateMode, save(modelPath(epoch), 'net', 'info') ; end
     
     
     figure(1);
     clf;
     hasError = 1;
     subplot(1,hasError+1,1);
     if ~evaluateMode
         semilogy(1:epoch,info.train.objective,'.--','linewidth',2);
     end
     grid on;
     h = legend(sets);
     set(h,'color','none');
     xlabel('training epoch');
     ylabel('cost value');
     title('objective');
     if hasError
         subplot(1,2,2);
         leg={};
         plot(1:epoch,info.val.acc','.--','linewidth',2);
         leg = horzcat(leg,strcat('val')); % ,opts.errorLabels
         set(legend(leg{:}),'color','none');
         grid on;
         xlabel('training epoch');
         ylabel('error');
         title('error')
     end
     drawnow;
     print(1,modelFigPath,'-dpdf');

end  
    
    
function [net,stats,StoreData] = process_epoch(opts, epoch, spd_train, trainInd, learningRate, net, StoreData)

training = learningRate > 0 ;
count1 = 0;

if training
    mode = 'training' ; 
else
    mode = 'validation' ; 
end

stats = [0 ; 0; 0; 0; 0] ;

numGpus = numel(opts.gpus) ;
if numGpus >= 1
    one = gpuArray(single(1)) ;
else
    one = single(1) ;
end

batchSize = opts.batchSize;
errors = 0;
numDone = 0 ;
flag = 0;

for ib = 1 : batchSize : length(trainInd) 
    flag = flag + 1;
    fprintf('%s: epoch %02d: batch %3d/%3d:', mode, epoch, ib,length(trainInd)) ;
    batchTime = tic ; 
    res = [];
    
    if (ib+batchSize> length(trainInd))
        batchSize_r = length(trainInd)-ib+1;  
    else
        batchSize_r = batchSize;
    end
    
    spd_data = cell(batchSize_r,1); % store the data in each batch    
    spd_label = zeros(batchSize_r,1); % store the label of the data in each batch
    
    for ib_r = 1 : batchSize_r
      
        spdPath = [spd_train.SpdDir '/' spd_train.spd.name{trainInd(ib+ib_r-1)}]; 
        load(spdPath);  % load the path of SPD data into the workspace
        spd_data{ib_r} = single(temp_2); % temp_2 denotes the name of each SPD data
        spd_label(ib_r) = spd_train.spd.label(trainInd(ib+ib_r-1));
        
    end
    
    net.layers{end}.class = spd_label; % label for cross-entropy loss
    net.layers{6}.class = spd_label; % label for metric learning regularizer
    
    %% forward/backward smdml
    if training
        dzdy = one; 
    else
        dzdy = [] ;
    end
    
    res = vl_myforbackward(net, spd_data, dzdy, res, epoch, count1) ; % forward computation and backward optimization
    
    % accumulating graidents
    if numGpus <= 1
      net = accumulate_gradients(opts, learningRate, batchSize_r, net, res) ;
    else
      if isempty(mmap)
        mmap = map_gradients(opts.memoryMapFile, net, res, numGpus) ;
      end
      write_gradients(mmap, net, res) ;
      labBarrier() ;
      net = accumulate_gradients(opts, learningRate, batchSize_r, net, res, mmap) ;
    end
          
    % accumulate training errors
    predictions = gather(res(end-1).x) ;
    [~,pre_label] = sort(predictions, 'descend') ;
    error = sum(~bsxfun(@eq, pre_label(1,:)', spd_label)) ;
    errors = errors+error;
    batchTime = toc(batchTime) ;
    speed = batchSize/batchTime ;
    numDone = numDone + batchSize_r ;
    
    stats = stats+[batchTime ; res(end).x; res(7).obj; res(13).obj; error]; %
    fprintf(' %.2f s (%.1f data/s)', batchTime, speed) ;
    fprintf(' l1-sm: %.5f', stats(2)/numDone) ;  % obj value of the classification term
    fprintf(' l2-ml: %.5f', stats(3)/numDone) ; % obj value of the metric learning term
    fprintf(' l2-rebud: %.5f', stats(4)/numDone) ; % obj value of the reconstruction error term
    fprintf(' l-mix: %.5f', (stats(2) + stats(3) + stats(4))/numDone) ;
    fprintf(' error: %.5f', stats(5)/numDone) ; % classification error
    fprintf(' [%d/%d]', numDone, batchSize_r);
    fprintf(' lr: %.6f',learningRate);
    fprintf('\n') ;  
    
end


% -------------------------------------------------------------------------
function [net,res] = accumulate_gradients(opts, lr, batchSize, net, res, mmap)
% -------------------------------------------------------------------------
for l = numel(net.layers):-1:1 
  if isempty(res(l).dzdw)==0 
    if ~isfield(net.layers{l}, 'learningRate')
       net.layers{l}.learningRate = 1 ;
    end
    if ~isfield(net.layers{l}, 'weightDecay')
       net.layers{l}.weightDecay = 1;
    end
    thisLR = lr * net.layers{l}.learningRate ;

    if isfield(net.layers{l}, 'weight')
        if strcmp(net.layers{l}.type,'bfc')==1
            W1 = net.layers{l}.weight;
            W1grad  = (1/batchSize)*res(l).dzdw;
            % gradient update on the Stiefel manifolds
            problemW1.M = stiefelfactory(size(W1,1), size(W1,2));
            W1Rgrad = (problemW1.M.egrad2rgrad(W1, W1grad)); 
            net.layers{l}.weight = (problemW1.M.retr(W1, -thisLR*W1Rgrad)); % retr indicates retraction (back onto manifold)
        else
            net.layers{l}.weight = net.layers{l}.weight - thisLR * (1/batchSize) * res(l).dzdw ;% update W_f of the FC layer
        end
    
    end
  end
end

