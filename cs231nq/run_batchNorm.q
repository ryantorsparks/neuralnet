\p 5000
\l load_all.q
cifarMode:`unflattened
\l load_cifar_data.q

lg "##############################
    Neural nets with batch normalization
    ##############################"

`. upsert `N`D1`D2`D3!200 50 60 3;
x:randArray[N;D1]
w1:randArray[D1;D2]
w2:randArray[D2;D3]
a:dot[(0|dot[x;w1]);w2]
lg "before batch norm, means and dev's of a are "
avg a
dev each flip a

lg "##############################
    Batchnorm forward
    ##############################"

lg "after batch norm (gamma=1, beta=0), mean and dev's are "
aNorm:first batchNormForward[a;D3#1f;D3#0f;enlist[`mode]!enlist`train] 
avg aNorm
dev each flip aNorm

lg "after batch norm (nontrivial gamma, beta)"
aNorm: first batchNormForward[a;1 2 3f;11 12 13f;enlist[`mode]!enlist `train]                                                                                        
avg aNorm
dev each flip aNorm

lg "check test time forward pass by running the training-time forward
    pass many times to warm up the running avgs, and then checking the
    means and variances of activations after a test-time forward pass"
`. upsert `N`D1`D2`D3!200 50 60 3;
w1:randArray[D1;D2]
w2:randArray[D2;D3]
bnParam:(1#`mode)!1#`train
gamma:D3#1f
beta:D3#0f
res:50{[bnParam;x;w1;w2;gamma;beta] last batchNormForward[dot[0f|dot[x;w1];w2];gamma;beta;bnParam]}[;x;w1;w2;gamma;beta]/bnParam 

x2:randArray[N;D1]
postRes:first batchNormForward[dot[0|dot[x2;w1];w2];gamma;beta;@[res;`mode;:;`test]] 
lg "after batch norm, mean and devs are"
avg postRes
dev each flip postRes

lg "##############################
    Batchnorm backward
    ##############################"

N:4
D:5
x:12+5*randArray[N;D]
gamma:first randArray[1;D]
beta:first randArray[1;D]
dout:randArray[N;D]

bnParam:(!). 1#'`mode`train
lg "get numerical grads"
dxNum:numericalGradientArray[(first batchNormForward[;gamma;beta;bnParam]@);x;dout;`x]
daNum:numericalGradientArray[(first batchNormForward[x;;beta;bnParam]@);gamma;dout;`gamma]
dbNum:numericalGradientArray[(first batchNormForward[x;gamma;;bnParam]@);beta;dout;`beta]

lg "now run batchNorm forward then backward"
cache:@[;1] batchNormForward[x;gamma;beta;bnParam]
dxDaDb: batchNormBackward[dout;cache]

lg "relative errors for dx, dgamma and dbeta are"
relError'[(dxNum;daNum;dbNum);dxDaDb]

lg "Alternative batchnorm backwards using derivs"
N:100
D:500
x:12+5*randArray[N;D]
gamma:first randArray[1;D]
beta:first randArray[1;D]
dout:randArray[N;D]

bnParam:(!). 1#'`mode`train
outCache:batchNormForward[x;gamma;beta;bnParam]

lg "run both batchnorm backwards versions 100 times and compare results/timings"
t1:system"t:100 res1:batchNormBackward[dout;outCache 1]"
t2:system"t:100 res2:batchNormBackwardAlt[dout;outCache 1]"
lg"dx, dgamma, dbeta relative differences are"
relError'[res1;res2]
lg"(time batchNormBackward)%time batchNormBackwardAlt was ",string t1%t2

lg "##############################
    Fully connected net with batchNorm
    ##############################"

`. upsert `N`D`H1`H2`C!2 15 20 30 10;
x:randArray[N;D]
y:N?C
startd:(!). flip ((`dimHidden;H1,H2);(`dimInput;D);(`nClass;C);(`wScale;5e-2);(`useBatchNorm;1b);(`useDropout;0b);(`x;x);(`y;y));
initd:fullyConnectedNet.init startd

lossGrad:fullyConnectedNet.loss initd;
lg "initial loss is ",string lossGrad 0

lg "as a sanity check, compare numerical gradients for reg in 0.0 3.14"
gradCheckDict:@[((raze key[startd],initd[`wParams`bParams`gammaParams`betaParams]),`wParams`bParams`gammaParams`betaParams`bnParams`L`flat)#initd;`model;:;`fullyConnectedNet]
compareNumericalGradients[gradCheckDict]each 0.0 3.14;


lg "##############################
    Fully connected net with batchNorm
    ##############################"

numTrain:1000
smallData:`xTrain`yTrain`xVal`yVal!(numTrain#xTrain;numTrain#yTrain;xVal;yVal)
startd:smallData,(!). flip (`model`fullyConnectedNet;(`dimHidden;5#100);(`nClass;10);(`wScale;2e-2);(`numEpocs;10);(`batchSize;50);(`updateRule;`adam);(`optimConfig;enlist[`learnRate]!enlist 1e-3);(`printEvery;200);(`learnRateDecay;0.95));

lg "run without batchnorm first"
res1:.solver.train @[startd;`useBatchNorm;:;0b]

lg "now run with batchnorm, should converge faster"
res2:.solver.train @[startd;`useBatchNorm;:;1b]

lg "plot results:
  
    scatter plot of:
    ([]iteration:til 200;lossNoBatch:res1`lossHistory;lossBatch:res2`lossHistory)
    line chart of: 
    ([]iteration:string til 1+count res1`trainAccHistory;trainAccNoBatch:0.,res1`trainAccHistory;trainAccBatch:0.,res2`trainAccHistory)
    line chart of: 
    ([]iteration:string til 1+count res1`valAccHistory;valAccNoBatch:0.,res1`valAccHistory;valAccBatch:0.,res2`valAccHistory)"

lg "##############################
    Batch norm and initialization
    ##############################"

numTrain:1000
smallData:`xTrain`yTrain`xVal`yVal!(numTrain#xTrain;numTrain#yTrain;xVal;yVal)
startd:smallData,(!). flip (`model`fullyConnectedNet;(`dimHidden;7#100);(`nClass;10);(`numEpocs;10);(`batchSize;50);(`updateRule;`adam);(`optimConfig;enlist[`learnRate]!enlist 1e-3);(`printEvery;200);(`learnRateDecay;0.95));

wScales:logSpace[-4;0;20]

lg "We now run training on a deep (7 hidden layer) network
    using batchnorm and non batchnorm, but we vary the weight 
    scale, using 20 different weightScales spread out 
    logarithmically between 1e-4 to 1, and then we plot results"

compareOneWeightScale:{[d;wScales;ind]
    wScale:wScales ind;
    lg "running ",string[ind],"/",string[count wScales]," training for non batchnorm, then batchnorm, for weight ",string wScale;
    d[`wScale]:wScale;
    aggs:(last;max;max);
    resKeys:`lossHistory`trainAccHistory`valAccHistory;
    res:`baseline,wScale,aggs@'.solver.train[@[d;`useBatchNorm;:;0b]]resKeys;
    res2:`batchNorm,wScale,aggs@'.solver.train[@[d;`useBatchNorm;:;1b]]resKeys;
    flip `method`wScale`finalLoss`bestTrainAcc`bestValAcc!flip (res;res2)
 }

res:raze compareOneWeightScale[startd;wScales;] each til count wScales
    
lg "now plot each result, ignoring any 0w's for finalLoss (from non batchnorm)"
res:update finalLoss:0n from res where finalLoss=0w
/ define pivot lambda, strings wScale for the line chart in qstudio
/ just input either `finalLoss, `bestTrainAcc or `bestValAcc
piv:{[res;col]update string wScale from (`wScale,`$string[cs],\:"_",string col)xcol ?[res;();{x!x}1#`wScale;(#;enlist cs:`baseline`batchNorm;(!;`method;col))]}[res;]

lg "use a line chart to plot:
    piv `finalLoss
    piv `bestTrainAcc
    piv `bestValAcc"



