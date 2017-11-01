/ ########## fullyConnectedNet class functions ###########

/ return list of params for a fully connected neural net given dict d
/ given a dict d, if it contains `modelParams already, return it,
/ otherwise if it has `wParams`bParams
.fullyConnectedNet.params:{[d]
    / d either expects `modelParams (exit early), both `wParams+`bParams (exit early),
    /     or it needs `dimHidden
    / if we already have it in d, return early
    if[`modelParams in key d;:d`modelParams];

    / if w and b params are in d, combine them and return
    if[all `wParams`bParams in key d;:raze d`wParams`bParams];

    / otherwise, make sure dimHidden is in d, and use that to create
    / if we have 5 hidden dimensions, model params will be `b1`b2...`b6`w1`w2...`w6
    if[not `dimHidden in key d;'".fullyConnectedNet.params: d is missing `dimHidden"];
    numLayers:1+count d`dimHidden;
    tnl:1+til numLayers;
    wParams:`$"w",/:string tnl;
    bParams:`$"b",/:string tnl;
    wParams,bParams
 };

.fullyConnectedNet.bnParams:{[d]
    if[all `gammaParams`beta in key d;:raze d`gammaParams`betaParams];
        
    / otherwise, make sure dimHidden is in d, and use that to create
    if[not `dimHidden in key d;'".fullyConnectedNet.params: d is missing `dimHidden"];
    numLayers:1+count d`dimHidden;
    tnl:1+til numLayers;
    gammaParams:`$"gamma",/:string tnl;
    betaParams:`$"beta",/:string tnl;
    gammaParams,betaParams
 };

/ get the layer inds (e.g. if we have 2 hidden layers, it's 1 2 3)
.fullyConnectedNet.layerInds:{[d]
    if[`layerInds in key d;:d`layerInds];

    if[`wParams in key d;:1+til count d`wParams];
    if[`bParams in key d;:1+til count d`bParams];

    if[`modelParams in key d;:1+til count[d`modelParams]div 2];

    if[`numLayers in key d;:1+til d`numLayers];

    if[not `dimHidden in key d;'".fullyConnectedNet.layerInds: needs `dimHidden"];
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
/ @global - sets .fullyConnectedNet.params here (list of `b1`b2`b3...`w1`w2`w3...
.fullyConnectedNet.init:{[d]
    / d expects at the very least `dimHidden
    defaults:(!) . flip (
        (`dimInput;3*32*32);
        (`nClass;10);
        (`dropout;0);
        (`useBatchNorm;0b);
        (`wScale;0.01);
        (`reg;0.0);
        (`flat;0b);
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
//  if[not all wParams in key d;d,:wParams!d[`wScale]*randArray ./:wDims];
//  if[not all wParams in key d;d,:wParams!.[;(::;1);d[`wScale]*]randArrayFlat ./:wDims];
    if[not all wParams in key d;d,:wParams!$[d`flat;.[;(::;1);d[`wScale]*]randArrayFlat ./:;d[`wScale]*randArray ./:]wDims];
    d[`bParams]:bParams;
    d[`wParams]:wParams;
    d[`layerInds]:.fullyConnectedNet.layerInds[d];
    d[`L`N`C]:(1+count d`dimHidden),d`dimInput`nClass;

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
.fullyConnectedNet.loss:{[d]
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
    modelParams:2 0N#.fullyConnectedNet.params d;
    wParams:modelParams 0;
    bParams:modelParams 1;
    layerInds:.fullyConnectedNet.layerInds d;


    / ####################### forward pass ##################
    / store everything in hidden dict
    hidden:()!();
//    hidden[`h0]:reshapeM[d`x;{x[0],prd 1_ x}shape d`x];
//    hidden[`h0]:({x[0],prd 1_ x}shape d`x;razeo d`x);
    hidden[`h0]:$[d`flat;({x[0],prd 1_ x}shape d`x;razeo d`x);reshapeM[d`x;{x[0],prd 1_ x}shape d`x]];

    if[d`useDropout;hidden[`hdrop0`cacheHdrop0]:dropoutForward[hidden`h0;d`dropoutParam]];

    / forward pass through all layers
    d:d[`L]fullyConnectedForwardPassLoop/@[d;`i`hidden;:;(0;hidden)];

    / scores, from last hidden h
    scores:d[`hidden]symi[`h;d`L];

    / exit early and return scores if we're doing test (not training)
    if[mode=`test;:scores];

    lossDscores:softmaxLoss `x`y!(scores;d`y);
    loss:lossDscores 0;
    dscores:lossDscores 1;

    / add on regularization for each  weights (sum of sum x*x for each weights)
//    loss+:0.5*d[`reg]*r$r:razeo d wParams;
//    loss+:0.5*d[`reg]*r$r:razeo d[wParams][;1];
    loss+:0.5*d[`reg]*r$r:razeo $[d`flat;.[;(::;1)];]d wParams;

    / ######################## backward pass ################
    
    d[`hidden;symi[`dh;d`L]]:dscores;
    d:d[`L]fullyConnectedBackwardPassLoop/@[d;`i;:;-1+d`L];
  
    / add on reg
//    d[`hidden;symi[`d;]each d`wParams]+:d[`reg]*d d`wParams;
//    d[`hidden;symi[`d;]each d`wParams;1]+:d[`reg]*last each d d`wParams;
    $[d`flat;
        d[`hidden;symi[`d;]each d`wParams;1]+:d[`reg]*last each d d`wParams;
        d[`hidden;symi[`d;]each d`wParams]+:d[`reg]*d d`wParams
     ];


    / grads should be `dw1`dw2..`db1`db2...`dbeta1`dbeta2...`dgamma1`dgamma2!...
    / TODO: make this cleaner
    grads:{(key[x] where key[x] like "d*")#x}d`hidden;
    / .solver.step expects `w1`w2 not `dw1`dw2 ..., so strip the d's
    / TODO: make this less hacky
    (loss;removeDFromDictKey grads)
 };

/ loop through forward passes incrementing d`i, and adding 
/ to d[`hidden] each time
fullyConnectedForwardPassLoop:{[d]
    idx:1+d`i;
    lastLayer:idx=d`L;
    / projection, param name eg p `a -> `a3 etc.
    p:symi[;idx];
    pe:p';
    w:d p`w;
    b:d p`b; 
    hidden:d`hidden;
    h:hidden symi[`h`hdrop@d`useDropout;idx-1];
    if[d[`useBatchNorm]&not lastLayer;
        gamma:d p`gamma;
        beta:d p`beta;
        bnParam:d[`bnParams;p`bnParam];
      ];
    / for last layer in forward pass, set h and cache h:
    if[lastLayer;hidden[pe`h`cacheH]:affineForward`x`w`b!(h;w;b)];

    / for all other layers
    if[not lastLayer;
        hCacheH:$[d`useBatchNorm;
                    affineNormReluForward `x`w`b`gamma`beta`bnParam!(h;w;b;gamma;beta;bnParam);
                    affineReluForward `x`w`b!(h;w;b)
                 ];
        hidden[pe`h`cacheH]:hCacheH;
        if[d`useDropout;hidden[pe`hdrop`cacheHdrop]:dropoutForward[hidden p`h;d`dropoutParam]];
      ];
    @[d;`hidden`i;:;(hidden;idx)]
 }; 

/ loop through backward pass, decrementing d`i
fullyConnectedBackwardPassLoop:{[d]
    idx:1+d`i;
    p:symi[;idx];
    pe:p';
    hidden:d`hidden;
    hCache:hidden p`cacheH;
    dh:hidden p`dh;
    lastLayer:idx=d`L;
    if[lastLayer; hidden[symi[`dh;idx-1],pe`dw`db]:affineBackward[dh;hCache]`dx`dw`db];
    if[not lastLayer;
        if[d`useDropout;dh:dropoutBackward[dh;hidden p`cacheHdrop]];
        $[d`useBatchNorm;
            hidden[symi[`dh;idx-1],pe`dw`db`dgamma`dbeta]:affineNormReluBackward[dh;hCache]`dx`dw`db`dgamma`dbeta;
            hidden[symi[`dh;idx-1],pe`dw`db]:affineReluBackward[dh;hCache]`dx`dw`db
         ];
      ];
    @[d;`hidden`i;:;(hidden;-1+d`i)]
 };

/ loss function for fully connected class
/ @param d: contains:
/ `w1`w2`w3 ... `b1`b2`b3  ... , `x and possibly `y
.old..fullyConnectedNet.loss:{[d]
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
    modelParams:2 0N#.fullyConnectedNet.params d;
    wParams:modelParams 0;
    bParams:modelParams 1;
    layerInds:.fullyConnectedNet.layerInds d;

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

    / add on regularization for each weights (sum of sum x*x for each weights)
    loss+:0.5*d[`reg]*r$r:razeo d wParams;
    
    / back prop into remaining layers
    / first do afineBackwards on final layer
    / indexes of all the layers, starting from 1, e.g. if we have `w1`w2...`w9, then
    / layerInds are 1 2 3 ... 9
    / add on reg to last dw (store as dict of `dxN`dwN`dbN)
    gradDict:renameGradKey[last layerInds;] affineBackward[dscores;cacheScores];
    gradDict:.fullyConnectedNet.backPropGrads[gradDict;1_ reverse layerInds;reverse caches;d`useBatchNorm];
    gradDict[wParams]+:d[`reg]*d wParams;
    (loss;gradDict)
 };


/ gradDict - is `dx`dw`db!(...), but also `dgamma`dbeta if we're doing batchNorm
/ revInds - e.g. for a net with input - hidden1 - hidden2 - hidden3 - output, 3 2 1
/ revCaches - list of caches corresponding to revInds, from afine(Norm)ReluForward
/ wlist - list of w's, e.g. (w3;w2;w1)
/ reg - regularization, e.g. 1e-5
.fullyConnectedNet.backPropGrads:{[gradDict;revInds;revCaches;useBatchNorm]
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
