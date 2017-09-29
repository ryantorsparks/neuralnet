/ ###################### n layer Convolution class #######################
/ An L-layer convolutional network with the following architecture:
/   [conv-relu-(2x2)pool]*L -> [affine-relu]*M -> affine -> softmax
/ The network operates on minibatches of data with shape (N;C;H;W),
/ with N images, each of height H and width W and with C input channels
/ possible inputs:
/        `dimInput: list (C, H, W) giving size of input data
/        `numFilters: List of ints/longs giving the number of filters in each
/          conv layer - count[numFilters] determines the number of conv layers L
/        `filterSize: Size of filters to use in the convolutional layers
/        `dimHidden: List of ints/longs - number of units to use in each
/          fully-connected hidden layer, count[dimHidden] determines number of hidden
/          layers M
/        `numClasses: Number of scores to produce from the final affine layer.
/        `wScale: Scalar giving standard deviation for random initialization
/          of weights.
/        `reg: Scalar giving L2 regularization strength
nLayerConvNet.init:{[d]
    / d expects at the very least `dimHidden
    defaults:(!) . flip (
        (`dimInput;3 32 32);
        (`numFilters;16 32);
        (`filterSize;3);
        (`dimHidden;100 100);
        (`nClass;10);
        (`useBatchNorm;0b);
        (`wScale;0.001);
        (`dropout;0f);
        (`seed;0N);
        (`dataAugmentation;1b);
        (`reg;0.0)
        );
    d:defaults,d;
    / set the number of conv layers L
    / and the number of hidden fully connected layers M
    d[`useDropout]:d[`dropout]>0;
    d[`L`M]:count each d`numFilters`dimHidden;

    / always set model to this
    d[`model]:`nLayerConvNet;
   
    / input dims are (C;H;W) as normal
    inputDims:d`dimInput;

    / conv layers, parameters of the conv is
    / F (# of filters), C,H,WW are size of each filter
    / Input size: N C H W
    / Output size: N F Hc Wc
    F:inputDims[0],d`numFilters;
    d[`F]:F;

    / add on W's, b's and bnParams (if in batchNorm)
    d:initWeightBiasBnParamsConvLayers[d];
    strideConv:1;

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

    / output height and widths of conv layers
    / inputDim[1 2] -> Hinput Winput (too many locals to actually define them)
    HConvWConv:sizeConv[strideConv;d`filterSize;inputDims 1;inputDims 2;d`L];

    / initialize the affine-relu layers
    dims:prd[HConvWConv,last F],d`dimHidden;
    if[not all{x=floor x}dims;'"dimensions must not be fractions"];
    d[`dims]:`long$dims;
    d:initWeightBiasBnParamsAffineReluLayers[d];

    / add W and b for the last layer, to d
    d[wp:`$"w",lastLayer:string sum 1,d`L`M]:d[`wScale]*rad last[dims],d`nClass;
    d[bp:`$"b",lastLayer]:d[`nClass]#0f;
    d[`wParams`bParams],:wp,bp;
    d
 };

nLayerFowardPassConvLayersLoop:{[d]
    / extract weights/bias relevant to this layer
    idx:d[`i]+1;
    sidx:string idx;
    w:d`$"w",sidx;
    b:d`$"b",sidx;

    / get the previous block
    h:d[`blocks]`$"h",string idx-1;
    
    / TODO possibly neaten this convoluted if else
    if[d`useBatchNorm;
        beta:d`$"beta",sidx;
        gamma:d`$"gamma",sidx;
        bnParam:d[`bnParams]`$"bnParam",sidx;
        hCacheH:convNormReluPoolForward[h;w;b;d`convParam;d`poolParam;gamma;beta;bnParam]
      ];
    if[not d`useBatchNorm;hCacheH:convReluPoolForward[h;w;b;d`convParam;d`poolParam]];
    h:hCacheH 0;
    cacheH:hCacheH 1;
    d[`blocks;`$"h",sidx]:h;
    d[`blocks;`$"cacheH",sidx]:cacheH;
   
    / increment i and return
    @[d;`i;:;idx]
 };

nLayerFowardPassLinearLayersLoop:{[d]
    / extract weights/bias relevant to this layer
    idx:sum 1,d`i`L;
    / projection, turn p`w -> `wn etc
    p:symi[;idx];
    pe:p';
    blocks:d`blocks;
    / if we're on the first fully connected layer, set dropout
    if[(0=d`i)&1b~d`useDropout;
        blocks[symi[;idx-1]each`hdrop`cacheHdrop]:dropoutForward[blocks@symi[`h;idx-1];d`dropoutParam]
      ];

    / get the previous block
    h:blocks symi[`h`hdrop@d`useDropout;idx-1];
    if[0=d`i;h:reshapeM[h;count[d`x],1_shape h]];

    / extract weight and bias
    w:d p`w;
    b:d p`b;

    / TODO possibly neaten this convoluted if else
    if[d`useBatchNorm;
        beta:d p`beta;
        gamma:d p`gamma;
        bnParam:d[`bnParams]p`bnParam;
        hCacheH:affineNormReluForward `x`w`b`gamma`beta`bnParam!(h;w;b;gamma;beta;bnParam);
      ];
    if[not d`useBatchNorm;hCacheH:affineReluForward `x`w`b!(h;w;b)];

    / add latest (h;cache) to blocks, i.e add `hN`cacheHN!hCacheH to d[`blocks]
    blocks[pe`h`cacheH]:hCacheH;
    if[d`useDropout;blocks[pe`hdrop`cacheHdrop]:dropoutForward[blocks p`h;d`dropoutParam]];

    / increment i and return
    @[d;`i`blocks;:;(1+d`i;blocks)]
 };

nLayerBackwardPassLinearLayersLoop:{[d]
    idx:sum 1,d`i`L;
    p:symi[;idx];
    pe:p';
    blocks:d`blocks;
    dh:blocks p`dh;
    if[d`useDropout;dh:dropoutBackward[dh;blocks p`cacheHdrop]];
    cacheH:blocks p`cacheH;
    / grads should be a dict `dx`dw`db[`dbeta`dgamma]!...
    if[d`useBatchNorm;
        grads:affineNormReluBackward[dh;cacheH];
        d[`blocks;pe`dbeta`dgamma]:grads`dbeta`dgamma;
      ];
    if[not d`useBatchNorm;grads:affineReluBackward[dh;cacheH]];

    / add in grads to blocks
    d[`blocks;symi[`dh;idx-1]]:grads`dx;
    d[`blocks;pe`dw`db]:grads`dw`db;
    @[d;`i;-;1]
 };
    
nLayerBackwardPassConvLayersLoop:{[d]
    idx:1+d`i;
    sidx:string idx;
    dh:d[`blocks]`$"dh",sidx;
    cacheH:d[`blocks]`$"cacheH",sidx;
    / for the furthest first (whiel moving from back to front in backward pass)
    / conv layer: we reshape using the +1'th layer's h
    if[d[`i]=d[`L]-1;
        dh:reshapeM[dh;shape d . `blocks,`$"h",sidx];
      ];
    / grads will be a dict `dx`dw`db[`dbeta`dgamma]!...
    if[d`useBatchNorm;
        grads:convNormReluPoolBackward[dh;cacheH];
        d[`blocks;`$("dbeta";"dgamma"),\:sidx]:grads`dbeta`dgamma;
      ];
    if[not d`useBatchNorm;grads:convReluPoolBackward[dh;cacheH]];

    / add in grads to blocks
    d[`blocks;`$"dh",string idx-1]:grads`dx;
    d[`blocks;`$("dw";"db"),\:sidx]:grads`dw`db;
    @[d;`i;-;1]
 };

/ loss function for n layer convnet, input/output same as threeLayerConvNet.loss
nLayerConvNet.loss:{[d]
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

    / pass convParam to the forward pass for the convolution layer
    d[`convParam]:`stride`pad!1,(d[`filterSize]-1)div 2;

    / add update mode of bnParams
    if[1b~d`useBatchNorm;
        d[`bnParams;::;`mode]:mode;
      ];

    / pass pool param to the foward pass for the max-pooling layer
    d[`poolParam]:`poolHeight`poolWidth`stride!3#2;

    / ####################### forward pass ##################
    / store dict of blocks
    blocks:()!();
    blocks[`h0]:d`x;
    d[`blocks]:blocks;

    / forward into the conv blocks
    d:d[`L]nLayerFowardPassConvLayersLoop/@[d;`i;:;0];
    
    / forward into linear blocks
    / add dropout (note only for fully connected, not conv layers)
    d:d[`M]nLayerFowardPassLinearLayersLoop/@[d;`i;:;0];

    / finally forward into score layer 
    idx:sum 1,d`L`M;
    p:symi[;idx];
    pe:p';
    w:d p`w;
    b:d p`b;
    h:d[`blocks]@symi[`h;idx-1];
    hCacheH:affineForward`x`w`b!(h;w;b);
    / add `hN`cacheHN!hCacheH to d[`blocks]
    d[`blocks;pe`h`cacheH]:hCacheH;

    / compute scores
    scores:d[`blocks]p`h;

    / exit early if we're in test mode (i.e. no y)
    if[not `y in key d;:scores];

    / ########### loss and scores ############
    / calc the loss
    lossDscores:softmaxLoss `x`y!(scores;d`y);
    dataLoss:lossDscores 0;
    dscores: lossDscores 1;
    loss:dataLoss+0.5*d[`reg]*r$r:razeo d d`wParams;

    / ########### backward pass ############
    idx:sum 1,d`L`M;
    p:symi[;idx];
    pe:p';
    dh:dscores;
    cacheH:d[`blocks]p`cacheH;
    dhDwDb:affineBackward[dh;cacheH];

    / add in grads for h/w/b from scoring layer
    d[`blocks;symi[`dh;idx-1]]:dhDwDb`dx;
    d[`blocks;wps:pe`dw`db]:dhDwDb`dw`db;
    d[`dwParams`dbParams],:wps;

    / now, backward pass for the linear layers/blocks
    d:d[`M]nLayerBackwardPassLinearLayersLoop/@[d;`i;:;-1+d`M];

    / now, backward pass for the linear layers/blocks
    d:d[`L]nLayerBackwardPassConvLayersLoop/@[d;`i;:;-1+d`L];

    / w grads where we add the reg. term
    / for every `dw[n] in d[`blcoks], add on d[`reg]*d[`dw[n]]
    d[`blocks;`$"d",/:string d`wParams]+:d[`reg]*d d`wParams;

    / grads should be `dw1`dw2..`db1`db2...`dbeta1`dbeta2...`dgamma1`dgamma2!...
    grads:raze[d`dwParams`dbParams`dgammaParams`dbetaParams]#d`blocks;
    / solver.step expects `w1`w2 not `dw1`dw2 ..., so strip the d's
    / TODO: make this less hacky
    (loss;removeDFromDictKey grads)
 };


/ initialize batchNorm params
nLayerConvNet.initBnParams:{[d;x;id;idOffset]
    lg "We use batchnorm here";
    gammas:x[id]#\:1f;
    betas:x[id]#\:0f;
    bnParams:`mode`runningMean`runningVar!(`train;x[id]#\:0f;x[id]#\:0f);
    ids:id+idOffset;
    kd:key d;
    / add in `beta0`beta1`dbeta0`dbeta1`betaParams`dbetaParams!(.....;`beta0`beta1;`dbeta0`dbeta1)
    d:@[d;`gammaParams`dgammaParams;(:;,)`gammaParams in kd;gp:`$("";"d"),/:\: "gamma",/:string ids];
    d:@[d;`betaParams`dbetaParams;(:;,)`betaParams in kd;bp:`$("";"d"),/:\: "beta",/:string ids];
    d[gp 0]:gammas;
    d[bp 0]:betas;
    d:@[d;`bnParams;(:;,)`bnParams in kd;([]bnParamName:`$"bnParam",/:string ids)!flip bnParams];
    d
 };

/ initialize weights and batch norm params for conv layers
/ d expects
/   L - long, number of conv layers
/   wScale - float, weight scale
/   F - long list - number of filters for each layer
/   filterSize - long - the number of filters, same all layers
/   useBatchNorm - boolean
/ TODO: make param name init less bad
initWeightBiasBnParamsConvLayers:{[d]
    l:til d`L;
    id:1+l;
    F:d`F;

    / init the weights
    ws:d[`wScale]*rad each F[l+\:1 0],\:2#d`filterSize;
    wParamNames:`$"w",/:string l+1;
    dwParamNames:`$"d",/:string wParamNames;
    d[`wParams`dwParams]:(wParamNames;dwParamNames);
    d:d,wParamNames!ws;

    / init biases
    bs:F[l+1]#\:0f;
    bParamNames:`$"b",/:string l+1;
    dbParamNames:`$"d",/:string bParamNames;
    d[`bParams`dbParams]:(bParamNames;dbParamNames);
    d:d,bParamNames!bs;

    / add in bn params
    if[d`useBatchNorm;d:nLayerConvNet.initBnParams[d;F;l+1;0]];
    d
 };


/ initialize weights and batch norm params for affine relu layers
/ d expects
/   L - long, number of conv layers
/   M - long, number of affine relu layers
/   wScale - float, weight scale
/   dims - long list - dims of affine relu layers
/   useBatchNorm - boolean
/ TODO: make param name init less bad
initWeightBiasBnParamsAffineReluLayers:{[d]
    m:til d`M;
    id:1+d[`L]+m;
    dims:d`dims;

    / initialize the list of w's (each will be 2 dims)
    ws:d[`wScale]*rad each dims m+\:0 1;
    wParamNames:`$"w",/:string id;
    dwParamNames:`$"d",/:string wParamNames;
    d[`wParams`dwParams],:(wParamNames;dwParamNames);
    d:d,wParamNames!ws;
   
    / init. the biases (each a list of 0's)
    bs:dims[m+1]#\:0f;
    bParamNames:`$"b",/:string id;
    dbParamNames:`$"d",/:string bParamNames;
    d[`bParams`dbParams],:(bParamNames;dbParamNames);
    d:d,bParamNames!bs;

    / add in bn params
    if[d`useBatchNorm;d:nLayerConvNet.initBnParams[d;dims;m+1;d`L]];
    d
 };

/ determine the conv size, iteratively (can also use over,
/ but think it's less clear)
sizeConv:{[strideConv;filterSize;H;W;nConv]
    / pad
    P:(filterSize-1)div 2;
    Hc:1+(H+(2*P)-filterSize)div strideConv;
    Wc:1+(W+(2*P)-filterSize)div strideConv;
    poolWidth:2;
    poolHeight:2;
    poolStride:2;
    Hp:1+(Hc-poolHeight)%poolStride;
    Wp:1+(Wc-poolWidth)%poolStride;
    $[nConv=1;(Hp;Wp);.z.s[strideConv;filterSize;Hp;Wp;nConv-1]]
 };

nLayerConvNet.params:{[d]raze d`wParams`bParams}
nLayerConvNet.bnParams:{[d]raze d`gammaParams`betaParams}

