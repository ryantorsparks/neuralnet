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
        (`reg;0.0)
        );
    d:defaults,d;
    / set the number of conv layers L
    / and the number of hidden fully connected layers M
    d[`L`M]:count each d`numFilters`dimHidden;

    / always set model to this
    d[`model]:`nLayerConvNet;
   
    / input dims are (C;H;W) as normal
    inputDims:d`dimInput;

    / conv layers, parameters of the conv is
    / F (# of filters), C,H,WW are size of each filter
    / Input size: N C H W
    / Output size: N F Hc Wc
    F:inputDims[0]+d`numFilters;



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
    / possibly use xavier init for wScale 
    / TODO: confirm this formula  is correct (fairly unsure)
    / fairly sure that the problems of bad initialisations are ameliorated
    / if we use batchnorm anyway, so probably not a big deal
    paramd[`w1]:rad[F,C,filterHeight,filterWidth]*
                    $[`xavier~d`weightFiller;
                        sqrt 2%1+C*filterHeight*filterWidth;
                        d`wScale
                     ];
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
    paramd[`w3]:convWInit[`weightFiller _ d;Hh,Hc;Hh+Hc];
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
    d:threeLayerConvNet.initBnParams[d;3];
    d
 };


