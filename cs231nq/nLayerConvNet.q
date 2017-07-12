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
    F:inputDims[0],d`numFilters;

    / add on W's, b's and bnParams (if in batchNorm)
    d:initWeightBiasBnParamsConvLayers[d];
    strideConv:1;

    / output height and widhts of conv layers
    / inputDim[1 2] -> Hinput Winput (too many locals)
    HConvWConv:sizeConv[strideConv;d`filterSize;inputDim 1;inputDim 2;d`L];

    / initialize the affine-relu layers
    dims:d[`dimHidden]+prd HConvWConv,last F;
    d:initWeightBiasBnParamsAffineReluLayers[d];

    / add W and b for the last layer, to d
    d[`$"W",lastLayer:string sum 1,d`L`M]:d[`wScale]*rad last[dims],d`nClass;
    d[`$"b",lastLayer]:d[`nClass]#0f;    
    d
 };


nLayerConvNet.initBnParams:{[d;x;id;idOffset]
    lg "We use batchnorm here";
    gammas:x[id]#\:1f;
    betas:x[id]#\:0f;
    bnParams:`mode`runningMean`runningVar!(`train;x[id]#\:0f;x[id]#\:0f);
    ids:id+idOffset;
    kd:key d;
    d:@[d;`gammaParams;(:;,)`gammaParams in kd;gp:`$"gamma",/:string ids];
    d:@[d;`betaParams;(:;,)`betaParams in kd;bp:`$"beta",/:string ids];
    d[gp]:gammas;
    d[bp]:betas;
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
initWeightBiasBnParamsConvLayers:{[d]
    l:til d`L;
    id:1+l;
    F:d`F;

    / init the weights
    Ws:d[`wScale]*rad each F[l+\:1 0],\:2#d`filterSize;
    d:d,(`$"W",/:string l+1)!Ws;

    / init biases
    bs:F[l+1]#\:0f;
    d:d,(`$"b",/:string l+1)!bs;

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
initWeightBiasBnParamsAffineReluLayers:{[d]
    m:til d`M;
    id:1+d[`L]+m;
    dims:d`dims;

    / initialize the list of W's (each will be 2 dims)
    Ws:d[`wScale]*rad each dims m+\:0 1;
    d:d,(`$"W",/:string id)!Ws;
   
    / init. the biases (each a list of 0's)
    bs:dims[m+1]#\:0f;
    d:d,(`$"b",/:string id)!bs;

    / add in bn params
    if[d`useBatchNorm;d:nLayerConvNet.initBnParams[d;dims;m+1;d`L]];
    d
 };

        


/ determine the conv size
sizeConv:{[strideConv;filterSize;H;W;nConv]
    / pad
    P:(filterSize-1)div 2;
    Hc:(H+(2*P)-filterSize)div strideConv+1;
    Wc:(W+(2*P)-filterSize)div strideConv+1;
    poolWidth:2;
    poolHeight:2;
    poolStride:2;
    Hp:(Hc-poolHeight)div poolStride+1;
    Wp:(Wc-poolWidth)div poolStride+1;
    $[nConv=1;(Hp;Wp);.z.s[strideConv;filterSize;Hp;Wp;nConv-1]]
 };












