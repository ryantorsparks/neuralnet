\l nn_util.q

affineForward:{[d]
    x:d`x;
    w:d`w;
    b:d`b;
    res:b+/:dot[(first[shape x],first shape w)#raze/[x];w];
    (res;`x`w`b!(x;w;b))
 };

affineBackward:{[dout;cached]
    x:cached `x;
    w:cached `w;
    b:cached `b;
    dw:dot[flip (first[shape x],first shape w)#raze/[x];dout];
    db:sum dout;
    dx:shape[x]#raze/[dot[dout;flip w]];
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

twoLayerNet.params:{[d]
    / use defaults if not provided
    temp::d;
    defaults:`dimInput`dimHidden`nClass`wScale`reg!(3*32*32;100;10;1e-3;0.0);
    d:defaults,d;
    b1:d[`dimHidden]#0.;
    w1:d[`wScale]*randArray . d`dimInput`dimHidden;
    b2:d[`nClass]#0.;
    w2:d[`wScale]*randArray . d`dimHidden`dimInput;
    d,`b1`w1`b2`w2!(b1;w1;b2;w2)
 };

/ @param d: contains:
/ `w1`w2`b1`b2`x and possibly `y
twoLayerNet.loss:{[d]
    / check if d has all necessary fields, if not then needs to do an init first
    if[not all `b1`w1`b2`w2 in key d;d:twoLayerNet.params d];

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

    regLoss:.5*d[`reg]*r$r:raze/[d`w1`w2];
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

/ ######## solver class functions ########
/ optional args:
/   `updateRule - e.g `sgd
/   `optimConfig - dict of hyperparams, each update rule needs different 
/       hyperparams, but all rules require a learnRate param
/   `learnRateDecay - after each epoch learnRate is multiplied by this
/   `batchSize - minibatch size for training
/   `epochs - number of epochs to run during training
/   `printEvery - training losses will be printery every printEvery iterations
solver.init:{[d]
    d:nulld,d;
    defaults:(!) . flip (
        (`updateRule;`sgd);
        (`optimConfig;nulld);
        (`optimConfigHistory;::);
        (`learnRateDecay;1.0);
        (`batchSize;100);
        (`epochs;10);
        (`printEvery;10)
        );
    d:defaults,d;
    if[not count key dur:d`updateRule;'"update rule `",string[dur]," not defined"];
 };

/ reset a bunch of dict variables
solver.reset:{[d]
    / book-keeping variables
    optimd:d`optimConfig;
    d,:(!) . flip (
        (`epoch;0);
        (`bestValAcc;0.0);
        (`bestParams;());
        (`lossHistory;());
        (`trainAccHistory;());
        (`valAccHistory;());
        / store optimConfig for each param in d
        (`optimConfigs;(`,key optimd)!(::),count[optimd]#enlist optimd)
        );
    d
 };

/ step function???
solver.step:{[d]
    / create mini batch
    numTrain:count d`xTrain;
    batchMask:d[`batchSize]?numTrain;
    xBatch:d[`xTrain] batchMask;
    yBatch:d[`yTrain] batchMask;

    / compute loss and grad of mini batch
    lossFunc:` sv d[`model],`loss;
    lossGrad:lossFunc `x`y!(d`xBatch;d`yBatch);
    loss:lossGrad 0;
    grads:lossGrad 1;
    d[`lossHistory],:loss;

    / parameter update
    {[d;p;w] 
        dw:grads p;
        config:d[`optimConfigs]p;
        nextWConfig:d[`updateRule][w;dw;config];
        nextW:nextWConfig 0;
        nextConfig:nextWConfig 1;
        d[p]:nextW;
        d[`optimConfigs;p]:nextConfig;
        d
    }/[d;
 };








