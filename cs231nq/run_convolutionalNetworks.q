@[system;"p 5000";{-1"WARNING: failed to set port to 5000";}]
\l load_all.q
\l load_cifar_data.q
/ set runAll to 1b only if you want to run everything
runAll:0b

lg "##############################
    Convolutional Networks
    ##############################"


lg "##############################
    Convolution: naive forward pass
    ##############################"

lg "The core of a convolutiona network is the convolution operation.
    Here we implement a naive (slow) forward pass"

xShape:2 3 4 4 
wShape:3 3 4 4 
x:xShape#linSpace[-0.1;0.5;prd xShape]
w:wShape#linSpace[-0.2;0.3;prd wShape]
b:linSpace[-0.1;0.2;3]
convParam:`stride`pad!2 1

out:first convForwardNaive[x;w;b;convParam]
lg "relative error with expected result is ",.Q.s1 relError[out;pget[`convNets;`expectedForwardOut]]

lg "##############################
    Convolution: naive backward pass
    ##############################"

lg "now we implement the backward pass for the convolution operation,
    then we do a numerical gradient check"

x:rad 4 3 5 5
w:rad 2 3 3 3
b:rad 2
dout:rad 4 2 5 5
convParam:`stride`pad!1 1

lg "first get numerical grads"
dxNum:numericalGradientArray[(first convForwardNaive[;w;b;convParam]@);x;dout;`x]
dwNum:numericalGradientArray[(first convForwardNaive[x;;b;convParam]@);w;dout;`w]
dbNum:numericalGradientArray[(first convForwardNaive[x;w;;convParam]@);b;dout;`b]

lg "now run batchNorm forward then backward"
outCache:convForwardNaive[x;w;b;convParam]
dxDwDb:convBackwardNaive[dout;outCache 1]

lg "numerical errors for `dx`dw`db are: "
relError'[value dxDwDb;(dxNum;dwNum;dbNum)]

lg "##############################
    Convolution: max pool naive forward
    ##############################"

lg "we now implement a naive forward pass for max pool layer"

xShape:2 3 4 4
x:xShape#linSpace[-0.3;0.4;prd xShape]
poolParam:`poolWidth`poolHeight`stride!2 2 2
outCache:maxPoolForwardNaive[x;poolParam]

lg "check relative error from expected output"
relError[outCache 0;pget[`convNets;`expectedForwardMaxPool]] 

lg "##############################
    Convolution: max pool naive backward
    ##############################"

lg "we now implement the naive backward pass for max pool layer"

x: rad 3 2 8 8 
dout:rad 3 2 4 4 
poolParam:`poolHeight`poolWidth`stride!2 2 2
outCache:maxPoolForwardNaive[x;poolParam]
dx:maxPoolBackwardNaive[dout;outCache 1]
dxNum:numericalGradientArray[(first maxPoolForwardNaive[;poolParam]@);x;dout;`x]

lg "check relative error against numerical gradient"
relError[dx;dxNum]

lg "##############################
    Convolutional 'sandwich' layers
    ##############################"

lg "##############################
    Testing conv relu pool forward
    ##############################"

x:rad 2 3 16 16
w:rad 3 3 3 3 
b:rad 3
dout:rad 2 3 8 8
convParam:`stride`pad!1 1
poolParam:`poolHeight`poolWidth`stride!2 2 2

outCache:convReluPoolForward[x;w;b;convParam;poolParam]
out:outCache 0;cache:outCache 1;
dxDwDb:convReluPoolBackward[dout;cache]

lg "now get numerical grads for comparison"
dxNum:numericalGradientArray[(first convReluPoolForward[;w;b;convParam;poolParam]@);x;dout;`x]
dwNum:numericalGradientArray[(first convReluPoolForward[x;;b;convParam;poolParam]@);w;dout;`w]
dbNum:numericalGradientArray[(first convReluPoolForward[x;w;;convParam;poolParam]@);b;dout;`b]
lg "check relative error against numerical gradient"
relError'[value dxDwDb;(dxNum;dwNum;dbNum)]

lg "##############################
    Testing conv relu forward
    ##############################"

x:rad 2 3 8 8
w:rad 3 3 3 3
b:rad 3
dout:rad 2 3 8 8
convParam:`stride`pad!1 1

outCache:convReluForward[x;w;b;convParam]
cache:outCache 1
dxDwDb:convReluBackward[dout;cache]

lg "now get numerical grads for comparison"
dxNum:numericalGradientArray[(first convReluForward[;w;b;convParam]@);x;dout;`x]
dwNum:numericalGradientArray[(first convReluForward[x;;b;convParam]@);w;dout;`w]
dbNum:numericalGradientArray[(first convReluForward[x;w;;convParam]@);b;dout;`b]

lg "check relative error against numerical gradient"
relError'[value dxDwDb;(dxNum;dwNum;dbNum)]

lg "##############################
    Three layer convnet
    ##############################"

x:rad 50 3 32 32
y:50?10
initd:threeLayerConvNet.init `x`y!(x;y)
lossGrad:threeLayerConvNet.loss initd
lg "initial loss (no regularization):"
lossGrad 0
lossGrad2:threeLayerConvNet.loss @[initd;`reg;:;0.5]
lg "initial loss (with regularization):"
lossGrad2 0

lg "##############################
    Gradient Check
    ##############################"

lg "After the loss looks reasonable, use numeric gradient checking to make sure 
    that your backward pass is correct. When you use numeric gradient checking 
    you should use a small amount of artifical data and a small number of 
    neurons at each layer"

numInputs:2
dimInput:3 16 16
reg:0.0
nClass:10
x:rad numInputs,dimInput
y:numInputs?nClass

startd:`numFilters`filterSize`dimInput`dimHidden`x`y!(3;3;dimInput;7;x;y)
initd: threeLayerConvNet.init startd
lossGrad:threeLayerConvNet.loss initd

lg "as a sanity check, compare numerical gradients for reg in 0.0 3.14"
gradCheckDict:@[(raze key[startd],`useBatchNorm`wScale`w1`w2`w3`b1`b2`b3)#initd;`model`h;:;`threeLayerConvNet,1e-6]
compareNumericalGradients[gradCheckDict;0f];

lg "##############################
    Overfit small data
    ##############################"

lg "A nice trick is to train your model with just a few training samples.
    You should be able to overfit small datasets, which will result in very 
    high training accuracy and comparatively low validation accuracy."

lg "first release some RAM with .Q.gc"
.Q.gc[]

lg "We start with only 100 data points"
numTrain:100
smallData:`xTrain`yTrain`xVal`yVal!(numTrain#xTrain;numTrain#yTrain;xVal;yVal)

lg "running 10 epochs of overfitting"
startd:smallData,(!). flip (`model`threeLayerConvNet;(`wScale;1e-2);(`numEpocs;10);(`batchSize;50);(`updateRule;`adam);(`optimConfig;enlist[`learnRate]!enlist 1e-3);(`printEvery;1));

if[runAll;res:solver.train startd]

lg "##############################
    Train the net
    ##############################"

lg "We now train a three layer convolutional network for one epoch.
    First call garbage collect"
.Q.gc[]

lg "##############################
    Base case
    ##############################"

lg "for the base case, should expect to see approx 20% accuracy"

startd:(!). flip ((`xTrain;xTrain);(`yTrain;yTrain);(`xVal;xVal);(`yVal;yVal);`model`threeLayerConvNet;(`wScale;1e-3);(`numEpochs;1);(`batchSize;50);(`updateRule;`adam);(`optimConfig;enlist[`learnRate]!enlist 1e-3);(`printEvery;20));

if[runAll;res:solver.train startd]

lg "##############################
    Decrease the filter size
    ##############################"

lg "instead of the default filter size of 7, use 3. Should get around 45-50% accuracy"

startd:(!). flip ((`xTrain;xTrain);(`yTrain;yTrain);(`xVal;xVal);(`yVal;yVal);`model`threeLayerConvNet;(`wScale;1e-3);(`numEpochs;1);(`batchSize;50);(`updateRule;`adam);(`optimConfig;enlist[`learnRate]!enlist 1e-3);(`printEvery;20);(`dimHidden;500);(`filterSize;3));

if[runAll;res:solver.train startd]

lg "##############################
    Experiment with spatial batchnorm
    ##############################"

lg "##############################
    Testing conv-norm-relu
    ##############################"

lg "first test out conv-norm-relu layers, by getting grads and then 
    comparing to numerical grads"

x:rad 2 3 16 16
w:rad 3 3 3 3
b:rad 3
gamma:3#1f
beta:3#1f
bnParam:`mode`runningMean`runningVar!(`train;3#0f;3#0f)
dout:rad 2 3 16 16
convParam:`stride`pad!2#1

outCache:convNormReluForward[x;w;b;convParam;gamma;beta;bnParam]
dxDwDbDgammaDbeta:convNormReluBackward[dout;outCache 1]

lg "now get numerical grads"
dxNum:numericalGradientArray[(first convNormReluForward[;w;b;convParam;gamma;beta;bnParam]@);x;dout;`x]
dwNum:numericalGradientArray[(first convNormReluForward[x;;b;convParam;gamma;beta;bnParam]@);w;dout;`w]
dbNum:numericalGradientArray[(first convNormReluForward[x;w;;convParam;gamma;beta;bnParam]@);b;dout;`b]
dgammaNum:numericalGradientArray[(first convNormReluForward[x;w;b;convParam;;beta;bnParam]@);gamma;dout;`x]
dbetaNum:numericalGradientArray[(first convNormReluForward[x;w;b;convParam;gamma;;bnParam]@);beta;dout;`beta]

lg "relative errors are"
lg relError'[(dxNum;dwNum;dbNum;dgammaNum;dbetaNum);dxDwDbDgammaDbeta]

lg "##############################
    Testing conv-norm-relu-pool
    ##############################"

lg "now, similar to just above, but we add in pool layer"

x:rad 2 3 16 16
w:rad 3 3 3 3
b:rad 3
gamma:3#1f
beta:3#1f
bnParam:`mode`runningMean`runningVar!(`train;3#0f;3#0f)
dout:rad 2 3 8 8
convParam:`stride`pad!2#1
poolParam:`poolHeight`poolWidth`stride!3#2

outCache:convNormReluPoolForward[x;w;b;convParam;poolParam;gamma;beta;bnParam]
dxDwDbDgammaDbeta:convNormReluPoolBackward[dout;outCache 1]

lg "now get numerical grads"
dxNum:numericalGradientArray[(first convNormReluPoolForward[;w;b;convParam;poolParam;gamma;beta;bnParam]@);x;dout;`x]
dwNum:numericalGradientArray[(first convNormReluPoolForward[x;;b;convParam;poolParam;gamma;beta;bnParam]@);w;dout;`w]
dbNum:numericalGradientArray[(first convNormReluPoolForward[x;w;;convParam;poolParam;gamma;beta;bnParam]@);b;dout;`b]
dgammaNum:numericalGradientArray[(first convNormReluPoolForward[x;w;b;convParam;poolParam;;beta;bnParam]@);gamma;dout;`x]
dbetaNum:numericalGradientArray[(first convNormReluPoolForward[x;w;b;convParam;poolParam;gamma;;bnParam]@);beta;dout;`beta]

lg "relative errors are"
lg relError'[(dxNum;dwNum;dbNum;dgammaNum;dbetaNum);dxDwDbDgammaDbeta]


lg "##############################
    Sanity check, loss with+w/o reg
    ##############################"

N:50
x:rad N,3 32 32
y:N?10

lg "run three layer loss with reg=0 then reg=0.5"

initd:threeLayerConvNet.init `x`y`useBatchNorm!(x;y;1b)
lossGrad:threeLayerConvNet.loss initd
lg "initial loss (no regularization):"
lossGrad 0
lossGrad2:threeLayerConvNet.loss @[initd;`reg;:;0.5]
lg "initial loss (with regularization):"
lossGrad2 0

lg "##############################
    Sanity check 2, grad check
    ##############################"


numInputs:2
dimInput:3 10 10
reg:0.0
nClass:10
x:rad numInputs,dimInput
y:numInputs?nClass

startd:`numFilters`filterSize`dimInput`dimHidden`x`y`useBatchNorm!(3;3;dimInput;7;x;y;1b)
initd: threeLayerConvNet.init startd
lossGrad:threeLayerConvNet.loss initd

lg "as a sanity check, compare numerical gradients for reg in 0.0 3.14"
gradCheckDict:@[(raze key[startd],`useBatchNorm`wScale`w1`w2`w3`b1`b2`b3`beta1`beta2`gamma1`gamma2`bnParams)#initd;`model`h;:;`threeLayerConvNet,1e-6] 
compareNumericalGradients[gradCheckDict;0f];

lg "##############################
    Sanity check 3, overfit using conv-relu-norm-pool
    ##############################"


lg "We start with only 100 data points"
numTrain:100
smallData:`xTrain`yTrain`xVal`yVal!(numTrain#xTrain;numTrain#yTrain;xVal;yVal)

lg "running 10 epochs of overfitting"
startd:smallData,(!). flip (`model`threeLayerConvNet;(`wScale;1e-2);(`numEpochs;10);(`batchSize;50);(`updateRule;`adam);(`optimConfig;enlist[`learnRate]!enlist 1e-3);(`printEvery;1);(`useBatchNorm;1b));

if[runAll;res:solver.train startd]

lg "##############################
    Long run of 3 layer net
    ##############################"

lg "this will take hours to run, and should get around 
    80% training acc., and >65% val. accuracy"

startd:(!). flip ((`xTrain;xTrain);(`yTrain;yTrain);(`xVal;xVal);(`yVal;yVal);`model`threeLayerConvNet;(`wScale;1e-3);(`numEpochs;4);(`batchSize;50);(`updateRule;`adam);(`optimConfig;enlist[`learnRate]!enlist 1e-3);(`printEvery;20);(`dimHidden;500);(`filterSize;3);(`useBatchNorm;1b));

if[runAll;res:solver.train startd]

lg "##############################
    Running a deeper conv net
    ##############################"

lg "##############################
    Sanity check, loss with+w/o reg
    ##############################"

N:50
x:rad N,3 32 32
y:N?10

lg "run 7 layer loss with reg=0 then reg=0.5"

initd:nLayerConvNet.init `x`y`useBatchNorm!(x;y;1b)
lossGrad:nLayerConvNet.loss initd
lg "initial loss (no regularization):"
lossGrad 0
lossGrad2:nLayerConvNet.loss @[initd;`reg;:;0.5]
lg "initial loss (with regularization):"
lossGrad2 0

lg "##############################
    Sanity check 2, nLayer convnet 
    grad check
    ##############################"

numInputs:2
dimInput:3 12 12
reg:0.0
nClass:10
x:rad numInputs,dimInput
y:numInputs?nClass

startd:`numFilters`filterSize`dimInput`dimHidden`x`y`useBatchNorm!(16 32;3;dimInput;6 6;x;y;1b)
initd: nLayerConvNet.init startd
lossGrad:nLayerConvNet.loss initd

lg "as a sanity check, compare numerical gradients for reg in 0.0 3.14"
gradCheckDict:@[(raze key[startd],`useBatchNorm`wScale`bnParams`L`M`wParams`dwParams`dbParams,raze(nLayerConvNet.params;nLayerConvNet.bnParams)@\:initd)#initd;`model`h;:;`nLayerConvNet,1e-6]
compareNumericalGradients[gradCheckDict;0f];

lg "##############################
    Sanity check 3, overfit using 
    n Layer convnet+small data
    ##############################"


lg "We start with only 100 data points"
numTrain:100
smallData:`xTrain`yTrain`xVal`yVal!(numTrain#xTrain;numTrain#yTrain;xVal;yVal)

lg "running 10 epochs of overfitting, larger convnet, 
    should reach 100% training accuracy within 10 epochs""
startd:smallData,(!). flip (`model`nLayerConvNet;(`wScale;1e-2);(`numEpochs;10);(`batchSize;50);(`updateRule;`adam);(`optimConfig;enlist[`learnRate]!enlist 1e-3);(`printEvery;1);(`useBatchNorm;1b);(`dimInputs;3 32 32);(`numFilters;16 32 64 128);`filterSize,3;(`dimHidden;500 500));

if[runAll;res:solver.train startd]

lg "##############################
    Run the full, deep conv net on
    all training data
    ##############################"

startd:(!). flip (`useBatchNorm,1b;(`numFilters;16 32 64 128);`batchSize,50;`updateRule`adam;`filterSize,3;`printEvery,10;(`dimHidden;500 500);(`dimInput;3 32 32);(`numEpochs;4);`wScale,.05;`learnRateDecay,0.95;`nClass,10;(`xTrain;xTrain);(`yTrain;yTrain);(`xVal;xVal);(`yVal;yVal);`model`nLayerConvNet;(`optimConfig;(enlist `learnRate)!enlist 1e-3);`reg,0.05)

if[runAll;res:solver.train startd]

lg "##############################
    Run a convnet on the mnist data
    ##############################"


startd:(!). flip (`useBatchNorm,1b;(`numFilters;32 32);`batchSize,50;`updateRule`adam;`filterSize,3;`printEvery,10;(`dimHidden;256 10);(`dimInput;1 28 28);(`numEpochs;4);`wScale,.05;`learnRateDecay,0.95;`nClass,10;(`xTrain;xTrain);(`yTrain;yTrain);(`xVal;xVal);(`yVal;yVal);`model`nLayerConvNet;(`optimConfig;(enlist `learnRate)!enlist 1e-3);`reg,0.05)

if[runAll;res:solver.train startd]
