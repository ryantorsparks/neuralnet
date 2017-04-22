\l nn_util.q
reshape:{[x;w](first[shape x],first shape w)#razeo x}

affineForward:{[d]
    x:d`x;
    w:d`w;
    b:d`b;
    res:b+/:dot[reshape[x;w];w];
    (res;`x`w`b!(x;w;b))
 };

affineBackward:{[dout;cached]
    x:cached `x;
    w:cached `w;
    b:cached `b;
    dw:dot[flip reshape[x;w];dout];
    db:sum dout;
    dx:shape[x]#razeo dot[dout;flip w];
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


/ ############ twoLayerNet class functions ############

/ learnable params
twoLayerNet.params:`w1`b1`w2`b2

twoLayerNet.init:{[d]
    / use defaults if not provided
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

    regLoss:.5*d[`reg]*r$r:razeo d`w1`w2;
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

/ ########## fullyConnectedNet class functions ###########

/ initialization/default params
/ possible inputs:
/ dimHidden - list of integers giving the size of each hidden layer
/ dimInput - integer giving size of the input
/ dropout - scalar b/w 0.0 and 1.0 giving dropout strength, 0=no dropout
/ useBatchNorm - boolean indicating to use batch normalization or not
/ reg - L2 regularization strength
/ weightScale - standard deviation for random initialization of weights
/ seed - if not none, then pass this random seed to dropout layers, which makes dropout layers
/        deterministic so we can gradient check the model
/ @global - sets fullyConnectedNet.params here (list of `b1`b2`b3...`w1`w2`w3...
fullyConnectedNet.init:{[d]
    defaults:(!) . flip (
        (`dimInput;3*32*32);
        (`nClass;10);
        (`dropout;0);
        (`useBatchNorm;0b);
        (`wScale;0.01);
        (`reg;0.0);
        (`seed;0N)
        );
    d:defaults,d;
    d[`useDropout]:d[`dropout]>0;
    numLayers:1+count d`dimHidden;
    d[`numLayers]:numLayers;

    / parameters of the network, w1 b1, w2 b2, etc.
    dims:raze d`dimInput`dimHidden`nClass;
 
    / set b1, b2, b3, ... etc., add to d
    bParams:`$"b",/:string tnl:1+til numLayers;
    d,:bParams!dims[tnl]#\:0f;
  
    / set w1, w2, w3, ... etc., where w1 has dimensions dims[0 1], 
    / w2 has dimensions dims[1 2], etc., add to d
    wDims:flip  1_'(prev dims;dims);
    wParams:`$"w",/:string tnl;
    d,:wParams!d[`wScale]*randArray ./:wDims;
    
    / now, also set fullyConnected.params (these are the params to learn)
    fullyConnectedNet.params:bParams,wParams;
    d[`bParams]:bParams;
    d[`wParams]:wParams;
    d[`layerInds]:tnl;
    lg "Set fullyConnectedNet.params as ",-3!fullyConnectedNet.params;

    / when using dropout, need to pass a dropoutParam dict to each dropout
    / layer so that the layer knows the dropout probability and the mode (train
    / vs. test). You can pass teh same dropoutParam to each dropout layer
    d[`dropoutParam]:()!();
    if[d`useDropout;
        d[`dropoutParam]:`mode`p!(`train;d`dropout);
        if[not null d`seed;
            d[`dropoutParam;`seed]:d`seed
          ]
      ];

    / for batch normalization, we need to keep track of running means and
    / variances, so need to pass a special bnParam object to each batch norm 
    / layer. so we use d[`bnParams;0] for the forward pass of the first batch
    / norm layer, and d[`bnParams;1] for the forward pass of the second batch
    / norm layer, etc.
    d[`bnParams]:$[d`useBatchNorm;
                     (numLayers-1)#enlist enlist[`mode]!enlist`train;
                     ()
                   ];
    d
 };

/ loss function for fully connected class
/ @param d: contains:
/ `w1`w2`w3 ... `b1`b2`b3  ... , `x and possibly `y
fullyConnectedNet.loss:{[d]
    mode:$[`y in key d;`train;`test];
    
    / set train test mode for batchnorm params and dropout param since they
    / behave differently during training and testing
    if[(()!())~d`dropoutParam;
        d[`dropoutParam;`mode]:mode
      ];

    / bnParam should have:
    /   mode - `train or `test
    /   eps: - constant for numerical stability
    /   momentum - constant for running mean/variance
    /   runningMean - array of shape (D,), running mean of features
    /   runningVar - shape (D,), running variance of features
    / ????? not sure about this update ?????
    if[1b~d`useBatchNorm;
        d:.[d;(`bnParams;mode);:;mode];
      ];

    / forward pass, compute class scores for x, store in scores
    / feed each first[outCache] (result of affineReluForward) into next affineReluForward
    / first, get wParams (`w1`w2`w3 ...`w[n-1]) and bParams (`b1`b2 ...`b[n-1])
    wParams:d`wParams;
    bParams:d`bParams;
    cacheLayers:{[outCache;w;b] affineReluForward @[d;`x`w`b;:;(outCache 0;w;b)]}\[(d`x;());d@-1_ wParams;d@-1_ bParams];
    layers: cacheLayers[;0];
    caches: cacheLayers[;1];

    / forward into last layer  
    scoresCache:affineForward @[d;`x`w`b;:;(last layers;d last wParams;d last bParams)];
    scores:scoresCache 0;
    cacheScores: scoresCache 1;

    / exit early and return scores if we're doing test (not training)
    if[mode=`test;:scores];

    lossDscores:softmaxLoss `x`y!(scores;d`y);
    loss:lossDscores 0;
    dscores:lossDscores 1;

    / add on regularization for each  weights (sum of sum x*x for each weights)
    loss+:0.5*d[`reg]*r$r:razeo d wParams;
    
    / back prop into remaining layers
    / first do afineBackwards on final layer
    / indexes of all the layers, starting from 1, e.g. if we have `w1`w2...`w9, then
    / layerInds are 1 2 3 ... 9
    layerInds:d`layerInds;
    dxDwDbTab:`layer xkey enlist @[affineBackward[dscores;cacheScores];`layer;:;last layerInds];
    / add on reg to last dw
    //dxDwDbTab[last layerInds;`dw]+:d[`reg]*d last wParams;
    dxDwDbDict:renameKey[last layerInds;] affineBackward[dscores;cacheScores];

    / backprop into remaining layers (cacheLayers from above)
    / each iteration uses the `dx from the previous iteration (i.e if we're currently
    / doing layer=7, it will use the `dx from layer 8)
    dxDwDbDict,:{[x;layer;cache;w;reg]
//        dxDwDb:affineReluBackward[x[layer+1;`dx];cache];
        dxDwDb:affineReluBackward[x[`$"x",string layer+1];cache];
        dxDwDb[`dw]+:reg*w;
        x,renameKey[layer;]dxDwDb
//        x upsert layer,dxDwDb`dx`dw`db
        }/[dxDwDbDict;1_ reverse layerInds;reverse caches;1_ reverse d wParams;d`reg];
//    (loss;dxDwDbTab)
    (loss;dxDwDbDict)
 };


/ ########### solver class functions ###########

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
   
    / add on initial default params for the model
    d:(` sv d[`model],`init)d;
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
    modelParams:value ` sv d[`model],`params;
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
    modelParams:value ` sv d[`model],`params;
  
    / ??? about the stuff after `reg
    lossGrad:lossFunc (inter[modelParams,`reg`dropoutParam`useBatchNorm`bnParams`wParams`bParams`layerInds;key d]#d),`x`y!(xBatch;yBatch);
    loss:lossGrad 0;
    grads:lossGrad 1;
    if[null loss;break];
    d[`lossHistory],:loss;

    / parameter update
    dchange:modelParams#d;
    d:{[d;p;w;grads] 
        //lg"updating parameter ",-3!p;
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
    yPred:raze {[f;d;x]{x?max x}peach f @[d;`x;:;x]}[lossFunc;`y _ d] peach x inds;

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
    res:numIterations solver.i.train/d;
    
    / finally, swap in best params
    res:res,res`bestParams;
    res
 };

/ called iteratively by sover.train
solver.i.train:{[d]
    / possibly print training loss
    cnt:d`cnt;
    numIterations:d`numIterations;

    / step
    d:solver.step d;

    if[0=cnt mod d`printEvery;
        lg"Iteration: ",string[d`cnt],"/",string[numIterations]," loss: ",string last d`lossHistory;
      ];

    / at end of every epoch, increment epoch counter, decay learnRate
    if[epochEnd:0=(1+cnt)mod d`iterationsPerEpoch;
        d[`epoch]+:1;
        d[`optimConfigs;;`learnRate]*:d`learnRateDecay;
      ];

    / check training and validation accuracy on first+last iteration,
    / and at the end of every epoch
    modelParams:value ` sv d[`model],`params;
    if[any (cnt=0;cnt=numIterations+1;epochEnd);
        trainAcc:solver.checkAccuracy (inter[modelParams,`bParams`wParams`layerInds;key d]#d),`model`x`y`batchSize`numSamples!d[`model`xTrain`yTrain`batchSize],1000;
        valAcc:solver.checkAccuracy (inter[modelParams,`bParams`wParams`layerInds;key d]#d),`model`x`y`batchSize`numSamples!d[`model`xVal`yVal`batchSize],0N;
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
    (w-dw*config`learnRate;config)
 }; 







