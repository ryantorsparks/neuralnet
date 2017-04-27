\l nn_util.q
reshape:{[x;w](first[shape x],first shape w)#razeo x}

affineForward:{[d]
    / d expects `x`w`b
    x:d`x;
    w:d`w;
    b:d`b;
    res:b+/:dot[reshape[x;w];w];
    (res;`x`w`b!(x;w;b))
 };

affineBackward:{[dout;cached]
    / cached expects `x`w`b
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

affineReluForward:{[d]
    / d should have `x`w`b (affineForward)
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
    / cache expects `x`w`b (affineBackward)
    fcCache:cache 0;
    reluCache:cache 1;
    da:reluBackward[dout;reluCache];
    dxDwDb:affineBackward[da;fcCache];
    dxDwDb
 };


/ ############ twoLayerNet class functions ############

/ learnable params, always  just these 4
twoLayerNet.params:{[d] `w1`b1`w2`b2}

/ layer inds, always 1 2
twoLayerNet.layerInds:{[d] 1 2}

twoLayerNet.init:{[d]
    / d expects nothing (defauls will provided for `dimInput`dimHidden`nClass`wScale`reg)
    / use defaults if not provided
    defaults:`dimInput`dimHidden`nClass`wScale`reg!(3*32*32;100;10;1e-3;0.0);
    d:defaults,d;
    b1:d[`dimHidden]#0.;
    w1:d[`wScale]*randArray . d`dimInput`dimHidden;
    b2:d[`nClass]#0.;
    w2:d[`wScale]*randArray . d`dimHidden`nClass;
   
    / always set model to `twoLayerNet
    d[`model]:`twoLayerNet;
    d,`b1`w1`b2`w2!(b1;w1;b2;w2)
 };

/ @param d: contains:
/ `w1`w2`b1`b2`x and possibly `y
twoLayerNet.loss:{[d]
    / d expects `x`w1`b1`w2`b2
    / d can also accept `y, and if provided (i.e. running train mode),
    /     then it expects `reg
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

/ return list of params for a fully connected neural net given dict d
/ given a dict d, if it contains `modelParams already, return it,
/ otherwise if it has `wParams`bParams
fullyConnectedNet.params:{[d]
    / d either expects `modelParams (exit early), both `wParams+`bParams (exit early),
    /     or it needs `dimHidden
    / if we already have it in d, return early
    if[`modelParams in key d;:d`modelParams];

    / if w and b params are in d, combine them and return
    if[all `wParams`bParams in key d;:raze d`wParams`bParams];

    / otherwise, make sure dimHidden is in d, and use that to create
    / if we have 5 hidden dimensions, model params will be `b1`b2...`b6`w1`w2...`w6
    if[not `dimHidden in key d;'"fullyConnectedNet.params: d is missing `dimHidden"];
    numLayers:1+count d`dimHidden;
    tnl:1+til numLayers;
    bParams:`$"b",/:string tnl;
    wParams:`$"w",/:string tnl;
    bParams,wParams
 };

/ get the layer inds (e.g. if we have 2 hidden layers, it's 1 2 3)
fullyConnectedNet.layerInds:{[d]
    if[`layerInds in key d;:d`layerInds];

    if[`wParams in key d;:1+til count d`wParams];
    if[`bParams in key d;:1+til count d`bParams];

    if[`modelParams in key d;:1+til count[d`modelParams]div 2];

    if[`numLayers in key d;:1+til d`numLayers];

    if[not `dimHidden in key d;'"fullyConnectedNet.layerInds: needs `dimHidden"];
    1+til 1+count d`dimHidden
 };

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
    / d expects at the very least `dimHidden
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
    
    / always set model to this
    d[`model]:`fullyConnectedNet;
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
    d[`bParams]:bParams;
    d[`wParams]:wParams;
    d[`layerInds]:fullyConnectedNet.layerInds[d];

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
    / d expects `dropoutParam`useBatchNorm`wParams(`w1`w2 ...`wN)`bParams(`b1`b2...`bN)
    /           `layerInds(1,2,3...N)
    / d possibly (???) needs `bnParams
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
    / test:
    modelParams:2 0N#fullyConnectedNet.params d;
    wParams:modelParams 0;
    bParams:modelParams 1;
    layerInds:fullyConnectedNet.layerInds d;

    / forward pass on all but last layer (scan and store result as caches)
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
    / add on reg to last dw (store as dict of `dxN`dwN`dbN)
    dxDwDbDict:renameKey[last layerInds;] affineBackward[dscores;cacheScores];

    / backprop into remaining layers (cacheLayers from above)
    / each iteration uses the `dx from the previous iteration (i.e if we're currently
    / doing layer=7, it will use the `dx from layer 8)
    dxDwDbDict,:{[x;layer;cache;w;reg]
        dxDwDb:affineReluBackward[x[`$"x",string layer+1];cache];
        dxDwDb[`dw]+:reg*w;
        x,renameKey[layer;]dxDwDb
        }/[dxDwDbDict;1_ reverse layerInds;reverse caches;1_ reverse d wParams;d`reg];
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
    / d expects `model (getModelValue)
    d:nulld,d;
   
    / add on initial default params for the model
    d:getModelValue[d;`init];
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
    / d expects `optimConfig`model
    / book-keeping variables
    optimd:d`optimConfig;

    / use [model].params[d] to get param list
    modelParams:getModelValue[d;`params];
    d,:(!) . flip (
        (`epoch;0);
        (`bestValAcc;0.0);
        (`bestParams;());
        (`lossHistory;());
        (`trainAccHistory;());
        (`valAccHistory;());
        / store optimConfig for each param in d
        (`optimConfigs;([]p:modelParams)!count[modelParams]#enlist optimd)
        );
    d
 };

/ step function???
solver.step:{[d]
    / d expects `xTrain`yTrain`batchSize`lossHistory`optimConfigs`updateRule
    / possibly (???( 
    / create mini batch
    numTrain:count d`xTrain;
    batchMask:neg[d`batchSize]?numTrain;
    xBatch:d[`xTrain] batchMask;
    yBatch:d[`yTrain] batchMask;

    / compute loss and grad of mini batch
    modelParams:getModelValue[d;`params];
  
    / ??? about the stuff after `reg
    lossGrad:getModelValue[ (inter[modelParams,`reg`dropoutParam`useBatchNorm`bnParams`wParams`bParams`layerInds`model;key d]#d),`x`y!(xBatch;yBatch);`loss];
    loss:lossGrad 0;
    grads:lossGrad 1;
    if[null loss;break];
    d[`lossHistory],:loss;

    / parameter update
    dchange:modelParams#d;
    d:{[d;p;w;grads] 
        dw:grads p;
        config:d[`optimConfigs]p;
        nextWConfig:d[`updateRule][w;dw;config];
        nextW:nextWConfig 0;
        nextConfig:nextWConfig 1;
        d[p]:nextW;
        optimConfigs:d`optimConfigs;
        if[count missingConfigParams:key[nextConfig] except cols optimConfigs;
            d[`optimConfigs]:![optimConfigs;();0b;missingConfigParams!count[missingConfigParams]#enlist (::)];
          ];
        d[`optimConfigs;p]:nextConfig;
        d
    }[;;;grads]/[d;key dchange;value dchange];
    d
 };

/ accuracy check
/ d is `x`y`numSamples`batchSize
solver.checkAccuracy:{[d]
    / d expects `x`y`model`numSamples`batchSize
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
    / d expects `xTrain`batchSize`numEpochs
    / first initialize d (this will first call model specific d[`model].init func, then
    / fill in blanks with default values)
    d: solver.init d;

    / then reset everything
    d: solver.reset d;

    / get # of trainings, numEpochs, iters per epoch etc.
    numTrain:count d`xTrain;
    iterationsPerEpoch:1|numTrain div d`batchSize;
    numIterations:d[`numEpochs]*iterationsPerEpoch;
    d[`numIterations`iterationsPerEpoch]:numIterations,iterationsPerEpoch;
    res:numIterations solver.i.train/d;
    
    / finally, swap in best params (only the model params though, leave
    / the rest in tact (e.g. loss/accuracy histories)
    res:res,getModelValue[res;`params]#res`bestParams;
    res
 };

/ called iteratively by sover.train
solver.i.train:{[d]
    / d expects `cnt`numIterations`printEvery`lossHistory`iterationsPerEpoch`epoch
    /           `optimConfigs`learnRate`learnRateDecay`model`batchSize`xTrain`yTrain
    /           `trainAccHistory`valAccHistory`bestValAcc
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
    modelParams:getModelValue[d;`params];
    if[any (cnt=0;cnt=numIterations+1;epochEnd);
        trainAcc:solver.checkAccuracy (inter[modelParams,`bParams`wParams`layerInds;key d]#d),`model`x`y`batchSize`numSamples!d[`model`xTrain`yTrain`batchSize],1000;
        valAcc:solver.checkAccuracy (inter[modelParams,`bParams`wParams`layerInds;key d]#d),`model`x`y`batchSize`numSamples!d[`model`xVal`yVal`batchSize],0N;
        d[`trainAccHistory],:trainAcc;
        d[`valAccHistory],:valAcc;
        lg"Epoch: ",string[d`epoch],"/",string[d`numEpochs]," train acc: ",string[trainAcc]," val acc: ",string[valAcc];
        
        / keep track of the best model
        if[valAcc>d`bestValAcc;
            d[`bestValAcc]:valAcc;
            d[`bestParams]:getModelValue[d;`params]#d;
          ];
      ];
    d[`cnt]+:1;
    d
 };


/ ########### optimiser (update rule) funcs #############

/ vanilla socastic gradient descent
sgd:{[w;dw;config]
    / config expects `learnRate
    (w-dw*config`learnRate;config)
 };

/ sgd with momentum
/ config is :
/   learnRate - float
/   momentum - foat within (0 1f) giving momentum value, 0 -> plain sgd
/   velocity - float array same shape as w and dw used to store a moving
/              average of the gradients
sgdMomentum:{[w;dw;config]
    / config expects `learnRate`momentum`velocity
    config:where[config~\:(::)] _ config;
    defaults:`learnRate`momentum!0.01 0.9;
    config:defaults,config;
    
    / velocity, default to 0's of shape w
    v:$[(`velocity in key config)and not all null razeo (),config`velocity;config`velocity;w*0.0];

    / momentum update
    v: (v*config`momentum)-dw*config`learnRate;
    config:config,enlist[`velocity]!enlist v;
    (w+v;config)
 };

/ rmsProp update rule, uses mavg of square gradient rules set adaptive per-
/ parameter learning rates
/ config format should be:
/   learnRate - float
/   updateDecayRate - float between 0 and 1f, decay rate of the squared gradient cache
/   epsilon - small float, used for smoothing to avoid dividing by 0
/   cache - mavg of second moments of gradients
rmsProp:{[x;dx;config]
    / d optionals `learnRate`updateDecayRate`epsilon`cache (defaults provided)
    defaults: `learnRate`updateDecayRate`epsilon`cache!(0.01;0.99;1e-8;x*0.0);

    / remove the null initialized ones (replace with default)
    config:where[config~\:(::)] _ config;
    config:defaults,config;
    
    / store next value of x as nextX
    cache:config`cache;
    updateDecayRate:config`updateDecayRate;
    epsilon:config`epsilon;
    learnRate:config`learnRate;
    cache:(cache*updateDecayRate)+(1-updateDecayRate)*dx*dx;
    nextX:x-learnRate*dx%epsilon+sqrt cache;
    config[`cache]:cache;
    (nextX;config)
 };

/ adam update, which incorporates moving averages of both the gradient
/ and its square, and a bias correction term
/ config format should be:
/   learnRate - float
/   beta1 - decay rate for mavg of first moment of gradient
/   beta2 - decay rate for mavg of second moment of gradient
/   epsilon - small float for smoothing to avoid dividing by 0
/   m - moving avg of gradient
/   v - moving average of squared gradient
/   t - iteration number
adam:{[x;dx;config]
    / d optionals `learnRate`beta1`beta2`epsilon`m`v`t
    defaults:(!) . flip ((`learnRate;1e-3);(`beta1;0.9);(`beta2;0.999);
             (`epsilon;1e-8);(`m;0f*x);(`v;0f*x);(`t;0));

    / remove the null initialized ones (replace with default)
    config:where[config~\:(::)] _ config;
    config:defaults,config;
    learnRate:config`learnRate;
    beta1:config`beta1;
    beta2:config`beta2;
    epsilon:config`epsilon;
    m:config`m;
    v:config`v;
    t:1+config`t;
    m:(beta1*m)+dx*1-beta1;
    v:(beta2*v)+(1-beta2)*dx*dx;
    
    / bias correction
    mb:m%1-beta1 xexp t;
    vb:v%1-beta2 xexp t;
    nextX:x-learnRate* mb % epsilon+sqrt vb;
    config[`m`v`t]:(m;v;t);
    (nextX;config)
 };







