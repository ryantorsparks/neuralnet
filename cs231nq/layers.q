/ functions for fully connected neural nets

affineForward:{[d]
    / d expects `x`w`b
    x:d`x;
    w:d`w;
    b:d`b;
    res:b+/:dot[reshape[x;w];w];
    (res;`x`w`b!(x;w;b))
 };

affineBackward:{[dout;cached]
    / cached expects `x`w`b
    x:cached `x;
    w:cached `w;
    b:cached `b;
    dw:dot[flipReshape[x;w];dout];
    db:sum dout;
    dx:reshapeM[dot[dout;flip w];shape x];
    `dx`dw`db!(dx;dw;db)
 };

reluForward:{(0.|x;x)}

reluBackward:{[dout;cache]
    dout*not cache<0
 };

affineReluForward:{[d]
    / d should have `x`w`b (affineForward)
    res:affineForward d;
    a:res 0;
    fcCache:res 1;
    res:reluForward[a];
    out:res 0;
    reluCache:res 1;
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
    da:reluBackward[dout;reluCache];
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


