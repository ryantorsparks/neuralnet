/ functions for fully connected neural nets

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
    dw:dot[flipReshape[x;w];dout];
    db:sum dout;
    dx:reshapeM[dot[dout;flip w];shape x];
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

/ used when we're doing batchNorm
affineNormReluForward:{[d]
    res:affineForward d;
    a:res 0;
    fcCache: res 1;
    bnParam:d`bnParam;
    resCache:batchNormForward[res 0;d`gamma;d`beta;d`bnParam];
    bnRes:resCache 0;
    bnCache:resCache 1;
    res:reluForward[bnRes];
    out:res 0;
    reluCache:res 1;
    cache:(fcCache;bnCache;reluCache);
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

affineNormReluBackward:{[dout;cache]
    fcCache:cache 0;
    bnCache:cache 1;
    reluCache:cache 2;
    reluRes:reluBackward[dout;reluCache];
    dxnDgammaDbeta:batchNormBackwardAlt[reluRes;bnCache];
    dxDwDb:affineBackward[dxnDgammaDbeta `dx;fcCache];
    / (dx;dw;db;dgamma;dbeta)
    dxDwDb,`dx _ dxnDgammaDbeta
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
    wParams:`$"w",/:string tnl;
    bParams:`$"b",/:string tnl;
    wParams,bParams
 };

fullyConnectedNet.bnParams:{[d]
    if[all `gammaParams`beta in key d;:raze d`gammaParams`betaParams];
        
    / otherwise, make sure dimHidden is in d, and use that to create
    if[not `dimHidden in key d;'"fullyConnectedNet.params: d is missing `dimHidden"];
    numLayers:1+count d`dimHidden;
    tnl:1+til numLayers;
    gammaParams:`$"gamma",/:string tnl;
    betaParams:`$"beta",/:string tnl;
    gammaParams,betaParams
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

    / only add w params if we don't have them already (may want to set them)
    if[not all wParams in key d;d,:wParams!d[`wScale]*randArray ./:wDims];
    d[`bParams]:bParams;
    d[`wParams]:wParams;
    d[`layerInds]:fullyConnectedNet.layerInds[d];

    / when using dropout, need to pass a dropoutParam dict to each dropout
    / layer so that the layer knows the dropout probability and the mode (train
    / vs. test). You can pass the same dropoutParam to each dropout layer
    d[`dropoutParam]:()!();
    if[d`useDropout;
        d[`dropoutParam]:`mode`p!(`train;d`dropout);
        if[not null d`seed;
            d[`dropoutParam;`seed]:d`seed
          ]
      ];

    / for batch normalization, we need to keep track of running means and
    / variances, so need to pass a special bnParam object to each batch norm 
    / layer. so we use d[`bnParams;] for the forward pass of the first batch
    / norm layer, and d[`bnParams;1] for the forward pass of the second batch
    / norm layer, etc.
    / TODO: use the initBnParams function
    if[d`useBatchNorm;        
        bnLayers:-1_tnl;
        bnZeros:(-1_1_dims)#\:0f;
        d[`gammaParams]:`$"gamma",/:string bnLayers;
        d[`betaParams]:`$"beta",/:string bnLayers;
        d[d`gammaParams]:1+bnZeros;
        d[d`betaParams]:bnZeros;
        d[`bnParams]:([bnParamName:`$"bnParam",/:string bnLayers] mode:`train;runningMean:bnZeros;runningVar:bnZeros);
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
    / if we have y, then treat this as training
    mode:`test`train@`y in key d;
 
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
        d:.[d;(`bnParams;::;`mode);:;mode];
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
    cacheLayers:
        $[d`useBatchNorm;
            cacheLayers:{[d;outCache;w;b;gamma;beta;bnParam] affineNormReluForward @[d;`x`w`b`gamma`beta`bnParam;:;(outCache 0;w;b;gamma;beta;bnParam)]}[d]\[(d`x;());d@-1_ wParams;d@-1_ bParams;d d`gammaParams;d d`betaParams;d `bnParams];
            cacheLayers:{[d;outCache;w;b] affineReluForward @[d;`x`w`b;:;(outCache 0;w;b)]}[d]\[(d`x;());d@-1_ wParams;d@-1_ bParams]
         ];
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
    gradDict:renameGradKey[last layerInds;] affineBackward[dscores;cacheScores];
    gradDict:fullyConnectedNet.backPropGrads[gradDict;1_ reverse layerInds;reverse caches;d`useBatchNorm];
    gradDict[wParams]+:d[`reg]*d wParams;
    (loss;gradDict)
 };

/ gradDict - is `dx`dw`db!(...), but also `dgamma`dbeta if we're doing batchNorm
/ revInds - e.g. for a net with input - hidden1 - hidden2 - hidden3 - output, 3 2 1
/ revCaches - list of caches corresponding to revInds, from afine(Norm)ReluForward
/ wlist - list of w's, e.g. (w3;w2;w1)
/ reg - regularization, e.g. 1e-5
fullyConnectedNet.backPropGrads:{[gradDict;revInds;revCaches;useBatchNorm]
    / backprop into remaining layers (cacheLayers from above)
    / each iteration uses the `dx from the previous iteration (i.e if we're currently
    / doing layer=7, it will use the `dx from layer 8)
    gradDict,:{[x;layer;cache;useBatchNorm]
        gradDict:$[useBatchNorm;affineNormReluBackward;affineReluBackward][x[`$"x",string layer+1];cache];
        x,renameGradKey[layer;]gradDict
        }/[gradDict;revInds;revCaches;useBatchNorm];

    / for non batchnorm, it's
    /     `dx`dw`db!(...)
    / for batchnnorm, it's
    /     `dx`dw`db`dgamma`dbeta
    gradDict
 };

/ ###################### Convolution class #######################
/ possible inputs:
/        `dimInput: list (C, H, W) giving size of input data
/        `numFilters: Number of filters to use in the convolutional layer
/        `filterSize: Size of filters to use in the convolutional layer
/        `dimHidden: Number of units to use in the fully-connected hidden layer
/        `numClasses: Number of scores to produce from the final affine layer.
/        `wScale: Scalar giving standard deviation for random initialization
/          of weights.
/        `reg: Scalar giving L2 regularization strength
threeLayerConvNet.init:{[d]
    / d expects at the very least `dimHidden
    defaults:(!) . flip (
        (`dimInput;3 32 32);
        (`numFilters;32);
        (`filterSize;7);
        (`dimHidden;100);
        (`nClass;10);
        (`useBatchNorm;0b);
        (`wScale;0.001);
        (`reg;0.0)
        );
    d:defaults,d;

    / always set model to this
    d[`model]:`threeLayerConvNet;
    inputDims:d`dimInput;
    C:inputDims 0;
    H:inputDims 1;
    W:inputDims 2;

    / conv layer, parameters of the conv is
    / F (# of filters), C,H,WW are size of each filter
    / Input size: N C H W
    / Output size: N F Hc Wc
    F:d`numFilters;
    filterHeight:d`filterSize;
    filterWidth:filterHeight;
    strideConv:1;

    / padding
    P:(d[`filterSize]-1)div 2;

    / output of convolution, height and width
    Hc:1+(H+(2*P)-filterHeight)div strideConv;
    Wc:1+(W+(2*P)-filterWidth)div strideConv;
    
    / initialise random w1, and b1
    paramd:()!();
    paramd[`w1]:d[`wScale]*rad F,C,filterHeight,filterWidth;
    paramd[`b1]:F#0f;


    / Pool layer, 2x2, pool layer has no params but is important in the
    / count of dimension:
    / Input: N F Hc Wc
    / Output: N F Hp Wp
    widthPool:heightPool:stridePool:2;
    Hp:1+(Hc-heightPool)div stridePool;
    Wp:1+(Wc-widthPool)div stridePool;

    / Hidden affine layer
    / size of parameter (F*Hp*Wp;H1)
    Hh:d`dimHidden;
    paramd[`w2]:d[`wScale]*rad (F*Hp*Wp;Hh);
    paramd[`b2]:Hh#0f;

    / output affine layer
    Hc:d`nClass;
    paramd[`w3]:d[`wScale]*rad Hh,Hc;
    paramd[`b3]:Hc#0f;

    / add to dicitonary d, but don't overwrite any values of W/b if they 
    / already exists (not likely at all, just for testing)
    d:paramd,d;

    / this should add these to d:
    / `bnParams - a keyed table of ([bnParamName] mode;runningMean;runningVar)
    /     where bnParamName should be bnParam1, bnParam2
    / `gammaParams - `gamma1`gamma2
    / d[`gamma1`gamm2] - d[`dimHidden`nClass]#\:1f
    / d[`betaParams] - `beta1`beta2
    / d[`beta1`beta2] - d[`dimHidden`nClass]#\:0f
    d:initBnParams[d;3];
    d
 };

/ loss function for three layer convnet, input/output same as twoLayerNet.loss
threeLayerConvNet.loss:{[d]
    / d expects `dropoutParam`useBatchNorm`wParams(`w1`w2 ...`wN)`bParams(`b1`b2...`bN)
    /           `layerInds(1,2,3...N)
    / d possibly (???) needs `bnParams
    / if we have y, then treat this as training
    mode:`test`train@`y in key d;
    if[1b~d`useBatchNorm;
        d:.[d;(`bnParams;::;`mode);:;mode];
      ];

    / pass convParam to the forward pass for the convolution layer
    filterSize:shape[d`w1]2;
    convParam:`stride`pad!1,(filterSize-1)div 2;

    / pass pool param to the foward pass for the max-pooling layer
    poolParam:`poolHeight`poolWidth`stride!3#2;

    / forward into the conv layer
    convCache:$[d`useBatchNorm;
                  convNormReluPoolForward[d`x;d`w1;d`b1;convParam;poolParam;d`gamma1;d`beta1;d`bnParam1];
                  convReluPoolForward[d`x;d`w1;d`b1;convParam;poolParam]
               ];
    convLayer:convCache 0;
    cacheConvLayer:convCache 1;

    / output shape
    convShape:shape convLayer;

    / forward pass into hidden layera
    / rshape x
    x:reshapeM[convLayer;(convShape 0;prd convShape 1 2 3)];
    hiddenCache:$[d`useBatchNorm;
                    affineNormReluForward `x`w`b`gamma`beta`bnParam!(x;d`w2;d`b2;d`gamma2;d`beta2;d`bnParam2);
                    affineReluForward`x`w`b!(x;d`w2;d`b2)
                 ];
    hiddenLayer:hiddenCache 0;
    cacheHiddenLayer:hiddenCache 1;
    /N:count hiddenLayer;
    /Hh:count first hiddenLayer;

    / forward into linear output layer
    scoresCache:affineForward`x`w`b!(hiddenLayer;d`w3;d`b3);
    scores:scoresCache 0;
    cacheScores:scoresCache 1;

    / exit early if we're in test mode (i.e. no y)
    if[not `y in key d;:scores];

    / ########### backward pass ############
    lossDscores:softmaxLoss `x`y!(scores;d`y);
    dataLoss:lossDscores 0;
    dscores: lossDscores 1;
    regLoss:.5*d[`reg]*r$r:razeo d`w1`w2`w3;
    loss:dataLoss+regLoss;

    / back propagation into output layer
    grads:renameGradKey[3;] affineBackward[dscores;cacheScores];
    grads[`w3]+:d[`reg]*d`w3;

    / backprop into first layer
    / both return `dx`dw`db, batchnorm also has `dbeta`dgamma
    grads,:renameGradKey[2;] $[d`useBatchNorm;affineNormReluBackward;affineReluBackward][grads`x3;cacheHiddenLayer];
    grads[`w2]+:d[`reg]*d`w2;

    / finally, backprop into conv layer
    / same return as grads2
    grads,:renameGradKey[1;] $[d`useBatchNorm;convNormReluPoolBackward;convReluPoolBackward][reshapeM[grads`x2;convShape];cacheConvLayer];
    grads[`w1]+:d[`reg]*d`w1;

    / `dx1`dw1`db1 .... `dx3`dw3`db3 (and possibly `beta1/2`gamma1/2)
    (loss;grads)
 };

threeLayerConvNet.params:{[d] `w1`b1`w2`b2`w3`b3}










/ for batch normalization, we need to keep track of running means and
/ variances, so need to pass a special bnParam object to each batch norm
/ layer. so we use d[`bnParams;] for the forward pass of the first batch
/ norm layer, and d[`bnParams;1] for the forward pass of the second batch
initBnParams:{[d;numLayers]
   if[d`useBatchNorm;
        lg "We use batchnorm here";
        if[not all (req:`dimInput`dimHidden`nClass) in key d;'"initBnParam: d needs all of ",-3!req];
        dims:raze d`dimInput`dimHidden`nClass;
        tnl:1+til numLayers;
        bnLayers:-1_tnl;
        bnZeros:(-1_1_dims)#\:0f;
        d[`gammaParams]:`$"gamma",/:string bnLayers;
        d[`betaParams]:`$"beta",/:string bnLayers;
        d[d`gammaParams]:1+bnZeros;
        d[d`betaParams]:bnZeros;
        d[`bnParams]:([bnParamName:`$"bnParam",/:string bnLayers] mode:`train;runningMean:bnZeros;runningVar:bnZeros);
      ];
   d
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
    if[d`useBatchNorm;modelParams,:raze d`betaParams`gammaParams];
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
    lossGradKeys:modelParams,`reg`dropoutParam`useBatchNorm`bnParams`wParams`bParams`betaParams`gammaParams`layerInds`model;
    if[d`useBatchNorm;lossGradKeys,:`gammaParams`betaParams,getModelValue[d;`bnParams]];
    lossGrad:getModelValue[ (inter[lossGradKeys;key d]#d),`x`y!(xBatch;yBatch);`loss];
    loss:lossGrad 0;
    grads:lossGrad 1;
    if[null loss;break];
    d[`lossHistory],:loss;

    / parameter update
    updateParams:modelParams,(();raze d`gammaParams`betaParams)d`useBatchNorm;
    dchange:updateParams#d;
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
//    batchSize:d`batchSize;
    batchSize:100;
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
        lg"checking accuracies";
        checkKeys:modelParams,`bParams`wParams`layerInds`useBatchNorm;
        if[d`useBatchNorm;checkKeys,:`bnParams`betaParams`gammaParams,getModelValue[d;`bnParams]];
        trainAcc:solver.checkAccuracy (inter[checkKeys;key d]#d),`model`x`y`batchSize`numSamples!d[`model`xTrain`yTrain`batchSize],1000;
        valAcc:solver.checkAccuracy (inter[checkKeys;key d]#d),`model`x`y`batchSize`numSamples!d[`model`xVal`yVal`batchSize],0N;
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

