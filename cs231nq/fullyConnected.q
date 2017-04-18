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





