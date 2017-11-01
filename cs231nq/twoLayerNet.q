/ ############ twoLayerNet class functions ############

/ learnable params, always  just these 4
.twoLayerNet.params:{[d] `w1`b1`w2`b2}

/ layer inds, always 1 2
.twoLayerNet.layerInds:{[d] 1 2}

.twoLayerNet.init:{[d]
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
.twoLayerNet.loss:{[d]
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

