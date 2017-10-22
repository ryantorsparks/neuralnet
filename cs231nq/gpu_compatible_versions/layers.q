/ functions for fully connected neural nets

affineForward:{[d]
    / d expects `x`w`b
    origx:d`x;
    shapex:origx 0;
    x:origx 1;
    origw:d`w;
    shapew:origw 0;
    w:origw 1;
    b:d`b;
//    res:b+/:dot[reshape[x;w];w];
    res:addDotBias[$[type b;b;b 1];dot[raze x;shapex 0;shapew 0;raze w;shapew 0;shapew 1];shapex 0;shapew 1];
    ((shapex[0],shapew 1;res);`x`w`b!(origx;origw;b))
 };

affineBackward:{[dout;cached]
    / cached expects `x`w`b
    origx:cached `x;
    shapex:origx 0;
    // turn n dim x -> 2 d version of x, e.g. shape[10 3 2 4]-> shape[10 12]
    shapexnew:shapex[0],prd 1_ shapex;
    x:origx 1;
    origw:cached `w;
    w:origw 1;
    shapew:origw 0;
    shapedout:dout 0;
    dout:dout 1;
    b:cached `b;
//    dw:dot[flipReshape[x;w];dout];
///   dw:dot[flip reshapeM[x;count[x],prd 1_ shape x];dout];
    dw:dot[flipFlat[x;shapexnew 0;shapexnew 1;prd[shapex]#0f];shapexnew 1;shapexnew 0;dout;shapedout 0;shapedout 1];
//    db:sum shapedout#dout;
    db:sumMatrixFlat[dout;shapedout 0;shapedout 1;shapedout[1]#0f];
    //dx:(reshapeM[dot[dout;flip w];shape x];
    dx:dot[dout;shapedout 0;shapedout 1;flipFlat[w;shapew 0;shapew 1;prd[shapew]#0f];shapew 1;shapew 0];
    `dx`dw`db!((shapex;dx);(shapew;dw);db)
 };

reluForward:{(0.|x;x)}

reluBackward:{[dout;cache]
    dout*not cache<0
 };

affineReluForward:{[d]
    / d should have `x`w`b (affineForward)
    res:affineForward d;
    / res is ((shaperes;res);cache)
    a:res 0;
    fcCache:res 1;
    res:enlist[a 0],reluForward[a 1];
    out:res 0 1;
    reluCache:res 2;
    cache:(fcCache;reluCache);
    (out;cache)
 };

/ used when we're doing batchNorm
affineNormReluForward:{[d]
    res:affineForward d;
    a:res 0;
    fcCache: res 1;
    bnParam:d`bnParam;
    resCache:batchNormForward[res 0;d`gamma;d`beta;d`bnParam];
    bnRes:resCache 0;
    bnCache:resCache 1;
    res:reluForward[bnRes];
    out:res 0;
    reluCache:res 1;
    cache:(fcCache;bnCache;reluCache);
    (out;cache)
 };

affineReluBackward:{[dout;cache]
    / cache expects `x`w`b (affineBackward)
    fcCache:cache 0;
    reluCache:cache 1;
    // dout is (shape dout;flat dout)
    da:(dout 0;reluBackward[dout 1;reluCache]);
    dxDwDb:affineBackward[da;fcCache];
    dxDwDb
 };

affineNormReluBackward:{[dout;cache]
    fcCache:cache 0;
    bnCache:cache 1;
    reluCache:cache 2;
    reluRes:reluBackward[dout;reluCache];
    dxnDgammaDbeta:batchNormBackwardAlt[reluRes;bnCache];
    dxDwDb:affineBackward[dxnDgammaDbeta `dx;fcCache];
    / (dx;dw;db;dgamma;dbeta)
    dxDwDb,`dx _ dxnDgammaDbeta
 };


