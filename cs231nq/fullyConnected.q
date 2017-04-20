\l nn_util.q

affineForward:{[d]
    x:d`x;
    w:d`w;
    b:d`b;
    res:b+/:dot[(first[shape x],first shape w)#raze/[x];w];
    (res;`x`w`b!(x;w;b))
 };

affineBackward:{[dout;cached]
    x:cached `x;
    w:cached `w;
    b:cached `b;
    dw:dot[flip (first[shape x],first shape w)#raze/[x];dout];
    db:sum dout;
    dx:shape[x]#raze/[dot[dout;flip w]];
    `dx`dw`db!(dx;dw;db)
 };

reluForward:{(0.|x;x)}

reluBackward:{[dout;cache]
    dout*not cache<0
 };

/ d should have `x`w`b
affineReluForward:{[d]
    res:affineForward d;
    a:res 0;
    fcCache:res 1;
    res:reluForward[a];
    out:res 0;
    reluCache:res 1;
    cache:(fcCache;reluCache);
    (out;cache)
 };

affineReluBackward:{[dout;cache]
    fcCache:cache 0;
    reluCache:cache 1;
    da:reluBackward[dout;reluCache];
    dxDwDb:affineBackward[da;fcCache];
    dxDwDb
 };

twoLayerNet.paramList:`w1`b1`w2`b2

twoLayerNet.params:{[d]
    / use defaults if not provided
    temp::d;
    defaults:`dimInput`dimHidden`nClass`wScale`reg!(3*32*32;100;10;1e-3;0.0);
    d:defaults,d;
    b1:d[`dimHidden]#0.;
    w1:d[`wScale]*randArray . d`dimInput`dimHidden;
    b2:d[`nClass]#0.;
    w2:d[`wScale]*randArray . d`dimHidden`nClass;
    d,`b1`w1`b2`w2!(b1;w1;b2;w2)
 };

/ @param d: contains:
/ `w1`w2`b1`b2`x and possibly `y
twoLayerNet.loss:{[d]
    / check if d has all necessary fields, if not then needs to do an init first
//    if[not all `b1`w1`b2`w2 in key d;show"not all wbs ";if[7=rand 10;break];d:twoLayerNet.params d];

    / forward into first layer
    hiddenCache:affineReluForward `x`w`b!d`x`w1`b1;
    hiddenLayer:hiddenCache 0;
    cacheHiddenLayer:hiddenCache 1;

    / forward into second layer
    scoresCache:affineForward `x`w`b!(hiddenLayer;d`w2;d`b2);
    scores:scoresCache 0;
    cacheScores:scoresCache 1;

    / if no y supplied, we're in test mode so return scores now
    if[not `y in key d;:scores];    

    / backward pass
    lossDscores:softmaxLoss `x`y!(scores;d`y);
    dataLoss:lossDscores 0;
    dscores: lossDscores 1;

    regLoss:.5*d[`reg]*r$r:raze/[d`w1`w2];
    loss:dataLoss+regLoss;

    / backprop into second layer
    dxwb:affineBackward[dscores;cacheScores];
    dx1:dxwb`dx;
    dw2:dxwb`dw;
    db2:dxwb`db;
    dw2+:d[`reg]*d`w2;
    
    / backprop into first layer
    dxwb:affineReluBackward[dx1;cacheHiddenLayer];
    dx:dxwb`dx;
    dw1:dxwb`dw;
    db1:dxwb`db;
    dw1+:d[`reg]*d`w1;
    grads:`w1`b1`w2`b2!(dw1;db1;dw2;db2);
    (loss;grads)
 };

/ ######## solver class functions ########
/ optional args:
/   `updateRule - e.g `sgd
/   `optimConfig - dict of hyperparams, each update rule needs different 
/       hyperparams, but all rules require a learnRate param
/   `learnRateDecay - after each epoch learnRate is multiplied by this
/   `batchSize - minibatch size for training
/   `numEpochs - number of epochs to run during training
/   `printEvery - training losses will be printery every printEvery iterations
solver.init:{[d]
    d:nulld,d;
   
    / add on init params for the model
    d:(` sv d[`model],`params)d;
    defaults:(!) . flip (
        (`cnt;0);
        (`updateRule;`sgd);
        (`optimConfig;()!());
        (`optimConfigHistory;::);
        (`learnRateDecay;1.0);
        (`batchSize;100);
        (`numEpochs;10);
        (`printEvery;10)
        );
    d:defaults,d;
    if[not count key dur:d`updateRule;'"update rule `",string[dur]," not defined"];
    d
 };

/ reset a bunch of dict variables
solver.reset:{[d]
    / book-keeping variables
    optimd:d`optimConfig;
    modelParams:value ` sv d[`model],`paramList;
    resetd::d;
    d,:(!) . flip (
        (`epoch;0);
        (`bestValAcc;0.0);
        (`bestParams;());
        (`lossHistory;());
        (`trainAccHistory;());
        (`valAccHistory;());
        / store optimConfig for each param in d
        (`optimConfigs;(modelParams)!count[modelParams]#enlist optimd)
        );
    d
 };

/ step function???
solver.step:{[d]
    / create mini batch
    numTrain:count d`xTrain;
    batchMask:neg[d`batchSize]?numTrain;
    xBatch:d[`xTrain] batchMask;
    yBatch:d[`yTrain] batchMask;

    / compute loss and grad of mini batch
    lossFunc:` sv d[`model],`loss;
    modelParams:value ` sv d[`model],`paramList;
    lossGrad:lossFunc (inter[modelParams,`reg;key d]#d),`x`y!(xBatch;yBatch);
    loss:lossGrad 0;
    grads:lossGrad 1;
    if[null loss;break];
    d[`lossHistory],:loss;

    / parameter update
    temp3::(d;loss;grads);
    dchange:modelParams#d;
    d:{[d;p;w;grads] 
        //lg"updating parameter ",-3!p;
        temp4::(d;p;w;grads);
        dw:grads p;
        config:d[`optimConfigs]p;
        nextWConfig:d[`updateRule][w;dw;config];
        nextW:nextWConfig 0;
        nextConfig:nextWConfig 1;
        d[p]:nextW;
        d[`optimConfigs;p]:nextConfig;
        d
    }[;;;grads]/[d;key dchange;value dchange];
    d
 };

/ accuracy check
/ d is `x`y`numSamples`batchSize
solver.checkAccuracy:{[d]
    / possibly sumbsample the data
    N:count d`x;
    batchSize:d`batchSize;
    x:d`x;
    y:d`y;
    if[(not null numSamples)&N>numSamples:d`numSamples;
        mask:neg[numSamples]?N;
        N:numSamples;
        x@:mask;
        y@:mask;
      ];
    numBatches:N div batchSize;

    / make sure we do at least enough batches
    if[not 0=N mod batchSize;
        numBatches+:1];

    / get indices of the individual batches, no overlaps
    inds:(batchSize*til numBatches)_til N;

    / get loss func, as it only has x, no y, should just return loss)
    lossFunc:` sv d[`model],`loss;

    / also get index of each max entry in resulting loss array
    yPred:raze {[f;d;x]{x?max x}peach f @[d;`x;:;x]}[lossFunc;`y _ d] peach d[`x] inds;

    / finally, return accuracy
    avg yPred=y
 };

/ train function
solver.train:{[d]
    / get # of trainings, numEpochs, iters per epoch etc.
    numTrain:count d`xTrain;
    iterationsPerEpoch:1|numTrain div d`batchSize;
    numIterations:d[`numEpochs]*iterationsPerEpoch;
    d[`numIterations`iterationsPerEpoch]:numIterations,iterationsPerEpoch;
    res:numIterations solver.trainInner/d;
    
    / finally, swap in best params
    res:res,res`bestParams;
    res
 };

/ called iteratively by sover.train
solver.trainInner:{[d]
    / possibly print training loss
    tempInner::d;
    cnt:d`cnt;
    numIterations:d`numIterations;

    / step
    d:solver.step d;

    if[0=cnt mod d`printEvery;
        temp1::(d;numIterations);
        lg"Iteration: ",string[d`cnt],"/",string[numIterations]," loss: ",string last d`lossHistory;
      ];

    / at end of every epoch, increment epoch counter, decay learnRate
    if[epochEnd:0=(1+cnt)mod d`iterationsPerEpoch;
        d[`epoch]+:1;
        d[`optimConfigs;;`learnRate]*:d`learnRateDecay;
      ];

    / check training and validation accuracy on first+last iteration,
    / and at the end of every epoch
    modelParams:value ` sv d[`model],`paramList;
    if[any (cnt=0;cnt=numIterations+1;epochEnd);
        trainAcc:solver.checkAccuracy (inter[modelParams;key d]#d),`model`x`y`batchSize`numSamples!d[`model`xTrain`yTrain`batchSize],1000;
        valAcc:solver.checkAccuracy (inter[modelParams;key d]#d),`model`x`y`batchSize`numSamples!d[`model`xVal`yVal`batchSize],0N;
        d[`trainAccHistory],:trainAcc;
        d[`valAccHistory],:valAcc;
        lg"Epoch: ",string[d`epoch],"/",string[d`numEpochs]," train acc: ",string[trainAcc]," val acc: ",string[valAcc];
        
        / keep track of the best model
        if[valAcc>d`bestValAcc;
            d[`bestValAcc]:valAcc;
            d[`bestParams]:`xTrain`x`xVal`yTrain`yVal`y _ d;
          ];
      ];
    d[`cnt]+:1;
    d
 };

/ vanilla socastic gradient descent
sgd:{[w;dw;config]
    (w-config[`learnRate]*dw;config)
 }; 







