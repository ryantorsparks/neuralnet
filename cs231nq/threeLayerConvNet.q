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
                  convNormReluPoolForward[d`x;d`w1;d`b1;convParam;poolParam;d`gamma1;d`beta1;d[`bnParams]`bnParam1];
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
                    affineNormReluForward `x`w`b`gamma`beta`bnParam!(x;d`w2;d`b2;d`gamma2;d`beta2;d[`bnParams]`bnParam2);
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
        bnZeros:(-1_ (count[d`dimInput]-1)_ dims)#\:0f;
        d[`gammaParams]:`$"gamma",/:string bnLayers;
        d[`betaParams]:`$"beta",/:string bnLayers;
        d[d`gammaParams]:1+bnZeros;
        d[d`betaParams]:bnZeros;
        d[`bnParams]:([bnParamName:`$"bnParam",/:string bnLayers] mode:`train;runningMean:bnZeros;runningVar:bnZeros);
      ];
   d
 };


