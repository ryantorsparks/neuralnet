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

    (`model`params!(twoLayerNetLoss;twoLayerNetParams))[f]@d
 };

twoLayerNetParams:{[d]
    / use defaults if not provided
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
twoLayerNetLoss:{[d]
    / forward into first layer
    hiddenCache:affineReluForward `x`w`b!d`x`w1`b1;
    hiddenLayer:hiddenCache 0;
    cacheHiddenLayer:hiddenCache 1;

    / forward into second layer
    scoresCache:affineForward `x`w`b!(hiddenOutCache[0];d`w2;d`b2);
    scores:scoresCache 0;
    cacheScores:scoresCache 1;

    / if no y supplied, we're in test mode so return scores now
    if[not `y in key d;:scores];    

    / backward pass
    lossScores:softmaxLoss `x`y!(scoresCache 0;d`y);
    dataLoss:lossScores 0;
    dscores: lossScores 1;

    regLoss:.5*d[`reg]*r$r:raze/[d`w1`w2];
    loss:dataLoss+regLoss;

    / backprop into second layer
    dxwb:affineBackward[dscores;cacheScores];
    dx1:dxwb 0;
    dw2:dxwb 1;
    db2:dxwb 2;
    dw2+:d[`reg]*d`w2;
    
    / backprop into first layer
    dxwb:affineReluBackward[dx1;cacheHiddenLayer];
    dx:dxwb 0;
    dw1:dxwb 1;
    db1:dxwb 2;
    dw1+:d[`reg]*d`w1;
    grads:`w1`b1`w2`b2!(dw1;db1;dw2;db2);
    (loss;grads)
 };

twoLayerNetModel:{[f;d]
    funcDict:`loss`params!(twoLayerNetLoss;twoLayerNetParams);
    if[not f in k:key funcDict;'"twoLayerNetModel needs one of ",(-3!k)," for f input"];
    funcDict[f]@d
 };






