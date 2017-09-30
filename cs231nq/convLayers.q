/ from http://cs231n.github.io/convolutional-networks/

/ weight initialization for convnets
/ if d[`weightFiller] is `xavier, then use xavier initialisation
convWInit:{[d;wDims;n]
    $[`xavier~d`weightFiller;
        [lg"using xavier init";rad[wDims]*sqrt 2%n];
        d[`wScale]*rad wDims
     ]
 };


/ pool function
/ @param m - matrix that we want to decrease
/ @param fSize - filter size
/ @param stride - stride size
/ e.g. pool[4 4#16?100;2;2]
pool:{[m;fSize;stride]
    n:count m 0;
    strides:1+(n-fSize)div stride;
    
    / index into m (i.e. each sector, flipped), then get max
    / this is typically much faster than trying to index into each sector
    (2#strides)#max razeo[m] (raze til[stride]+/:n*til stride)+\:raze (stride*til strides)+/:(n*stride)*til strides
 };

/ convolution function - forward pass
/ @param V - Volume (matrix, not padded yet, should be square)
/ @param F - Filter (matrix, should be square)
/ @param b - bias (list of floats, should be same as first[shape F])
/ @param padSize - number of zeros to pad volume matrix V with
/ @param stride - the stride length across padded V
/ e.g.
/ Shape:2 3 4 4
/ wShape:3 3 4 4
/ x:xShape#linSpace[-0.1;0.5;prd xShape]
/ w:wShape#linSpace[-0.2;0.3;prd wShape]
/ b:linSpace[-0.1;0.2;3]
/ convParam:`stride`pad!2 1
/ out:first convForwardNaive[x;w;b;convParam]
/ TO DO: MAKE THIS FUNCTION LESS CRAP/SLOW!!
convForwardNaive:{[x;w;b;convParam]
    stride:convParam`stride;
    pad:convParam`pad;
    xShape:shape x;
    N:xShape 0;C:xShape 1;H:xShape 2;W:xShape 3;
    wShape:shape w;
    F:wShape 0; HH:wShape 2;WW:wShape 3;
    hout:`long$1+(H+(2*pad)-HH)%stride;
    wout:`long$1+(W+(2*pad)-WW)%stride;
    outShape:N,F,hout,wout;

    / convolution each func (many layers deep):
    / d is `F`x`hout`wout`HH`WW`stride`pad!(F;x;hout;wout;HH;WWstride;pad)
    convInner:{[d]
        / ind is n
        {[d;ind]
            d[`convIn]:zeroPad[d[`x] ind;d`pad];
            / inds are (n;f)
            {[d;inds]
                d[`convW]:d[`w] last inds;
                d[`convB]:d[`b] last inds;
                / inds are (n;f;i)
                {[d;inds]
                    / inds are (n;f;i;j)
                    {[d;inds]
                        convI:inds[2]*d`stride;
                        convJ:inds[3]*d`stride;
                        convArea:.[d`convIn;(::;convI+til d`HH;convJ+til d`WW)];
                        d[`convB]+sumo convArea*d`convW
                    }[d;] each inds,/:til d`wout
                }[d;] each inds,/:til d`hout
            }[d;] each ind,/:til d`F
        }[d;]each til d`N
    };
    out:convInner `N`F`x`w`b`hout`wout`HH`WW`stride`pad!(N;F;x;w;b;hout;wout;HH;WW;stride;pad);
    cache:`x`w`b`convParam!(x;w;b;convParam);
    (out;cache)
 };

/ also known as, convForwardStrides
convForwardFast:{[x;w;b;convParam]
   
    / unpack variables
    stride:convParam`stride;
    pad:convParam`pad;
    xShape:shape x;
    N:xShape 0;C:xShape 1;H:xShape 2;W:xShape 3;
    wShape:shape w;
    F:wShape 0; HH:wShape 2;WW:wShape 3;
    hout:`long$1+(H+(2*pad)-HH)%stride;
    wout:`long$1+(W+(2*pad)-WW)%stride;
    outShape:N,F,hout,wout;

    / assumes x is 4 dimensions
    if[not 4= count xShape;'"convForwardFast requires x to be 4 dimensions"];
    xPad:.[x;(::;::);zeroPad[;pad]];
    H+:2*pad;
    W+:2*pad;
    outh:1+(H-HH)div stride;
    outw:1+(W-WW)div stride;
    
    / perform an im2col operation by picking clever strides
    strideShape:C,HH,WW,N,outh,outw;
    strides:(H*W; W; 1; C*H*W; stride*W;stride);
    xCols: asStrided[xPad;strideShape;strides];
    
    / reshape from a 6D into a 2D matrix
    xCols:reshapeM[xCols;(C*HH*WW;N*outh*outw)];

    / now all our convolutions become one big matrix multiply
    res:dot[reshapeM[w;(F;0N)];xCols]+b;

    / reshape the output
    out:flip reshapeM[res;(F;N;outh;outw)];

    cache:`x`w`b`convParam`xCols!(x;w;b;convParam;xCols);
    (out;cache)
 };


/ convolution function, backward pass
/ @param dout - upstream derivatives
/ @param cache - (x;w;b;convParam), second output of convForwardNaive
/ @return `dx`dw`db!(dx;dw;db) gradients with respect to x, w, b
convBackwardNaive:{[dout;cache]
    x:cache`x;
    w:cache`w;
    b:cache`b;
    convParam:cache`convParam;
    stride:convParam`stride;
    pad:convParam`pad;

    / shape of x - > (N;C;H;W)
    xShape:shape x;
    N:xShape 0; C:xShape 1; H:xShape 2;W:xShape 3;

    / shape of w -> (F;C;HH;WW)
    wShape:shape w;
    F:wShape 0; HH:wShape 2;WW:wShape 3;

    hout:`long$1+(H+(2*pad)-HH)%stride;
    wout:`long$1+(W+(2*pad)-WW)%stride;
    dx:shape[x]#0f;
    dw:shape[w]#0f;
    db:shape[b]#0f;
    
    / horrible overs, 4 levels deep
    convBackInner:{[d]
       / ind is n
       {[d;ind]
           pad:d`pad;
           d[`convIn]:zeroPad[d[`x] ind;pad];
           cShape:shape d`convIn;
           d[`dconvIn]:cShape#0f;
           / inds are (n;f)
          d:{[d;inds]
               d[`convW]:d[`w] last inds;
               d[`convB]:d[`b] last inds;
               d[`df]:d[`dout] . inds;
               / inds are (n;f;i)
               {[d;inds]
                   / inds are (n;f;i;j)
                   {[d;inds]
                       convI:inds[2]*d`stride;
                       convJ:inds[3]*d`stride;
                       convArea:.[d`convIn;(::;convI+til d`HH;convJ+til d`WW)];
                       dconv:d[`df] . inds 2 3;
                       d:.[d;(`db;inds 1);+;dconv];
                       d:.[d;(`dw;inds 1);+;dconv*convArea];
                       d:.[d;(`dconvIn;::;convI+til d`HH;convJ+til d`WW);+;dconv*d`convW];
                       d
                   }/[d;inds,/:til d`wout]
               }/[d;inds,/:til d`hout]
           }/[d;ind,/:til d`F];
           .[d;(`dx;ind);+;.[d[`dconvIn];(::),(pad _neg[pad]_til@)each 1_ cShape]] 
       }/[d;til d`N]
    };
   inputd:`dout`N`F`x`w`b`dx`dw`db`hout`wout`HH`WW`stride`pad!(dout;N;F;x;w;b;dx;dw;db;hout;wout;HH;WW;stride;pad);
   res:convBackInner inputd;
   `dx`dw`db#res
 };

maxPoolForwardNaive:{[x;poolParam]
    stride:poolParam`stride;
    HH:poolParam`poolHeight;
    WW:poolParam`poolWidth;
    xShape:shape x;
    N:xShape 0;
    C:xShape 1;
    H:xShape 2;
    W:xShape 3;
    hout:`long$1+(H-HH)%stride;
    wout:`long$1+(W-HH)%stride;
    
    poolForwardInner:{[d]
        {[d;ind]
            {[d;inds]
                {[d;inds]
                    / inds (n;c;i;j)
                    {[d;inds]
                        poolI:inds[2]*d`stride;
                        poolJ:inds[3]*d`stride;
                        maxo .[d`x;(inds 0;inds 1;poolI+til d`HH;poolJ+til d`WW)]
                    }[d;]each inds,/:til d`wout
                }[d;]each inds,/:til d`hout
            }[d;]each ind,/:til d`C
        }[d;]each til d`N
    };
   
   inputd:`x`N`C`hout`wout`stride`HH`WW!(x;N;C;hout;wout;stride;HH;WW);
   out:poolForwardInner inputd;
   cache:`x`poolParam!(x;poolParam);
   (out;cache)
 };

maxPoolForwardFast:{[x;poolParam]
    xShape:shape x;
    N:xShape 0;
    C:xShape 1;
    H:xShape 2;
    W:xShape 3;   
   
    sameSize:all 1_(~':)poolParam`poolHeight`poolWidth`stride;
    tiles:(0=H mod  poolParam`poolHeight) and 0=W mod poolParam`poolWidth;
    if[not sameSize and tiles;'"must have sameSize and tiles as true"];
/    if[r:sameSize and tiles;
        outReshapeCache:maxPoolForwardReshape[x;poolParam];
        reshapeCache:outReshapeCache 1;
        cache:`method`reshapeCache!(`reshape;reshapeCache);
        out:outReshapeCache 0; 
/      ];
/    if[not r;
/        outIm2colCache:maxPoolForwardIm2Col[x;poolParam];
/        out:outIm2colCache 0;
/        cache:(`im2col;outIm2colCache 1);
/      ];
    (out;cache)
 };

maxPoolForwardReshape:{[x;poolParam]
    xShape:shape x;
    N:xShape 0;
    C:xShape 1;
    H:xShape 2;
    W:xShape 3;
    poolHeight:poolParam`poolHeight;
    poolWidth:poolParam`poolWidth;
    stride:poolParam`stride;
    if[any 1_differ poolHeight,poolWidth,stride;'"pool params invalid"];
    xReshaped:reshapeM[x;(N;C;H div poolHeight;poolHeight;W div poolWidth;poolWidth)];
    out:maxAxes[xReshaped;3 4];
    cache:`x`xReshaped`out!(x;xReshaped;out);
    (out;cache)
 };

maxPoolBackwardFast:{[dout;cache]
    method:cache`method;
    realCache:cache`reshapeCache;
    if[not method~`reshape;'"pool method must be reshape"];
    / TODO: fix bug in maxPoolBackwardReshapeFast6D where it seg faults for realCache of count <= 4
    $[4<count realCache`xReshaped;maxPoolBackwardReshapeFast6D;maxPoolBackwardReshapeSlow][dout;realCache]
 };

maxPoolBackwardReshapeSlow:{[dout;cache]
    x:cache`x;
    xReshaped:cache`xReshaped;
    out:cache`out;

    dxReshaped:xReshaped*0f;
    outNewaxis:newAxes[out;3 5];
    mask:(=). broadcastArrays[xReshaped;outNewaxis];
    maskInds:where razeo mask;
    doutNewaxis:newAxes[dout;3 5];
    doutBroadcast: first broadcastArrays[doutNewaxis;dxReshaped];
    dxReshaped:shape[dxReshaped]#@[razeo dxReshaped;maskInds;:;razeo[doutBroadcast]@maskInds];
    broadcastRes:last broadcastArrays[dxReshaped;sumAxesKeepDims[mask;3 5]];
    dxReshaped%:broadcastRes;
    dx:reshapeM[dxReshaped;shape x];
    dx
 };

maxPoolBackwardReshapeFast6D:{[dout;cache]
    x:cache`x;
    xReshaped:cache`xReshaped;
    out:cache`out;

    dxReshaped:xReshaped*0f;
    outNewaxis:newAxes[out;3 5];
    floatMask:maskBroadcast6dAxes35[outNewaxis;xReshaped;shape outNewaxis;shape xReshaped];
    maskInds:where `boolean$razeo floatMask;
    doutNewaxis:newAxes[dout;3 5];
    doutBroadcast: first broadcastArrays[doutNewaxis;dxReshaped];
    dxReshaped:shape[dxReshaped]#@[razeo dxReshaped;maskInds;:;razeo[doutBroadcast]@maskInds];
    floatRes:sumAxes35KeepDims6dBroadcast[floatMask;shape floatMask];
    dxReshaped%:floatRes;
    dx:reshapeM[dxReshaped;shape x];
    dx
 };

/ currently ditched, python version uses cython/c, too much effort 
/ to translate to c, too slow to do in q (a million for loops), so
/ just going to force having pool height+width+stride the same
/
maxPoolForwardIm2Col:{[x;poolParam]
    xShape:shape x;
    N:xShape 0;
    C:xShape 1;
    H:xShape 2;
    W:xShape 3;
    poolHeight:poolParam`poolHeight;
    poolWidth:poolParam`poolWidth;
    stride:poolParam`stride;
    if[not 0=(H-poolHeight)mod stride;'"invalid pool height"];
    if[not 0=(W-poolWidth)mod stride;'"invalid pool width"];
    outHeight:1+(H-poolHeight)div stride;
    outWidth:1+(W-poolWidth)div stride;

    xSplit:(N*C;1;H;W)#x;
    xCols:
\


maxPoolBackwardNaive:{[dout;cache]
    stride:poolParam`stride;
    HH:poolParam`poolHeight;
    WW:poolParam`poolWidth;
    xShape:shape x;
    N:xShape 0;
    C:xShape 1;
    H:xShape 2;
    W:xShape 3;
    hout:`long$1+(H-HH)%stride;
    wout:`long$1+(W-HH)%stride;
    dx:xShape#0f;

    poolBackwardInner:{[d]
        {[d;ind]
            {[d;inds]
                {[d;inds]
                    / inds (n;c;i;j)
                    {[d;inds]
                        poolI:inds[2]*d`stride;
                        poolJ:inds[3]*d`stride;
                        poolArea:.[d`x;(inds 0;inds 1;poolI+til d`HH;poolJ+til d`WW)];
                        poolMax:maxo poolArea;
                        poolMaxMask:poolMax=poolArea;
                        .[d;(`dx;inds 0;inds 1;poolI+til d`HH;poolJ+til d`WW);+;poolMaxMask*dout . inds]
                    }/[d;inds,/:til d`wout]
                }/[d;inds,/:til d`hout]
            }/[d;ind,/:til d`C]
        }/[d;til d`N]
    };
    
    res:poolBackwardInner[`dx`x`N`C`hout`wout`stride`HH`WW!(dx;x;N;C;hout;wout;stride;HH;WW)];
    res`dx
 };

/ @param x - input array
/ @param w - weight param for conv layer
/ @param b - bias param
/ @param convParam - dict
convReluForward:{[x;w;b;convParam]
    aConvCache:convForwardFast[x;w;b;convParam];
    a:aConvCache 0;
    convCache:aConvCache 1;
    outReluCache:reluForward a;
    out:outReluCache 0;
    reluCache:outReluCache 1;
    cache:`convCache`reluCache!(convCache;reluCache);
    (out;cache)
 };

convReluForwardNaive:{[x;w;b;convParam]
    aConvCache:convForwardNaive[x;w;b;convParam];
    a:aConvCache 0;
    convCache:aConvCache 1;
    outReluCache:reluForward a;
    out:outReluCache 0;
    reluCache:outReluCache 1;
    cache:`convCache`reluCache!(convCache;reluCache);
    (out;cache)
 };

/ @param dout - array
/ @param cache - (convCache;reluCache)
/ @return `dx`dx`db!(grads ...)
convReluBackward:{[dout;cache]
    convCache:cache`convCache;
    reluCache:cache`reluCache;
    da:reluBackward[dout;reluCache];
    dxDwDb:convBackwardNaive[da;convCache];
    dxDwDb
 };

/ to be phased out
convReluPoolForwardNaive:{[x;w;b;convParam;poolParam]
    a_convCache:convForwardNaive[x;w;b;convParam];
    a:a_convCache 0;
    convCache:a_convCache 1;
    s_reluCache:reluForward[a];
    s:s_reluCache 0;
    reluCache:s_reluCache 1;
    out_poolCache:maxPoolForwardFast[s;poolParam];
    out:out_poolCache 0;
    poolCache:out_poolCache 1;
    cache:`convCache`reluCache`poolCache!(convCache;reluCache;poolCache);
    (out;cache)
    };

convReluPoolForward:{[x;w;b;convParam;poolParam]
    a_convCache:convForwardFast[x;w;b;convParam];
    a:a_convCache 0;
    convCache:a_convCache 1;
    s_reluCache:reluForward[a];
    s:s_reluCache 0;
    reluCache:s_reluCache 1;
    out_poolCache:maxPoolForwardFast[s;poolParam];
    out:out_poolCache 0;
    poolCache:out_poolCache 1;
    cache:`convCache`reluCache`poolCache!(convCache;reluCache;poolCache);
    (out;cache)
 };

/ to be phased out
convReluPoolBackwardNaive:{[dout;cache]
    convCache:cache`convCache;
    reluCache:cache`reluCache;
    poolCache:cache`poolCache;
    ds:maxPoolBackwardFast[dout;poolCache];
    da:reluBackward[ds;reluCache];
    dxDwDb:convBackwardNaive[da;convCache];
    dxDwDb
 };

/ fast version
convReluPoolBackward:{[dout;cache]
    convCache:cache`convCache;
    reluCache:cache`reluCache;
    poolCache:cache`poolCache;
    / suss
    ds:maxPoolBackwardFast[dout;poolCache];
    / correct
    da:reluBackward[ds;reluCache];
    dxDwDb:convBackwardFast[da;convCache];
    dxDwDb
 };


/ when doing conv backward fast, we need to initialize a few variables
/ that are very slow to create, but are used again and again
/ @params d - dict, should have `N`C`H`W`HH`WW`pad`stride`outh`outw
/ @example .conv.initBackwardVars[`N`C`H`W`HH`WW`pad`stride!50 3 32 32 7 7 3 1]
.conv.initBackwardVars:{[d]
    stride:d`stride;
    pad:d`pad;
    Hpad:d[`H]+2*pad;
    Wpad:d[`W]+2*pad;
    d[`outh`outw]:1+(Hpad-d`HH;Wpad-d`WW)div stride;
    inds:{raze x,/:\:y}/[til each d`N`C`HH`WW`outh`outw];
    .conv.colValInds:inds[;1 2 3 0 4 5];

    xpadInds:(inds[;0];inds[;1];inds[;2]+inds[;4]*stride;inds[;3]+stride*inds[;5]);
    .conv.gxpadInds:group flip xpadInds;

    .conv.padResIndOrder:key[.conv.gxpadInds]?cross/[til each (d`N;d`C;Hpad;Wpad)];
    .conv.padResDims:(d`N;d`C;Hpad;Wpad);
    .conv.finalIndex:(::;::;pad _ neg[pad]_ til Hpad;pad _ neg[pad] _ til Wpad);
 };
 
/ used by convReluPoolBackward
/ dout/da, cache/convCache
convBackwardFast:{[dout;cache]
    x:cache`x;
    w:cache`w;
    b:cache`b;
    convParam:cache`convParam;
    xCols:cache`xCols;
  
    xShape:shape x;
    N:xShape 0;
    C:xShape 1;
    H:xShape 2;
    W:xShape 3;  

    wShape:shape w;
    F:wShape 0;
    HH:wShape 2;   
    WW:wShape 3;
   
    doutShape:shape dout;
    outh:doutShape 2;
    outw:doutShape 3;

    db:sumAxes[dout;0 2 3];
    doutReshaped:reshapeM[flip dout;(F;0N)];
    /CHECK!!!
    dw:reshapeM[dot[doutReshaped;flip xCols];wShape];
    
    dxCols:dot[flip reshapeM[w;(F;0N)];doutReshaped];
    dxCols:reshapeM[dxCols;C,HH,WW,N,outh,outw];

    dx:col2im6d `xCols`stride`pad`H`HH`W`WW`N`C!(dxCols;convParam`stride;convParam`pad;H;HH;W;WW;N;C);
    `dx`dw`db!(dx;dw;db)
 };

col2im6d:{[d]
    stride:d`stride;
    pad:d`pad;
    Hpad:d[`H]+2*pad;
    Wpad:d[`W]+2*pad;
    outh:1+(Hpad-d`HH)div stride;
    outw:1+(Wpad-d`WW)div stride;
    xPaddedShape:(d`N;d`C;Hpad;Wpad);
    xCols:d`xCols;    
    col2im6dShape:d`C`HH`WW`N`H`W;

    / call out to c for this function, muuuch too slow in q unfortunately 
    xCols:reshapeM[xCols;col2im6dShape];
    res:col2im6dInner[xCols;xPaddedShape#0f;col2im6dShape;pad;stride];

    / if we've padded, index out
    if[pad>0;res:.[res;(::;::;pad _ neg[pad]_ til Hpad;pad _ neg[pad] _ til Wpad)]];
    res
 };

/ used by convBackwardFast, needs to have .conv.initBackwardVars run first
col2im6dOld:{[dxCols]   
    colVals:matrixDotInds[dxCols;.conv.colValInds];
    gRes:sum each colVals@.conv.gxpadInds;
    padResFlat:value[gRes]@.conv.padResIndOrder;
    padRes:.conv.padResDims#padResFlat;
    .[padRes;.conv.finalIndex] 
 };

/ convenience layer that performs a convolution, spatial batch norm,
/ relu
/ inputs:
/ x - input to convolution layer
/ w - weight list for conv layer
/ b - bias list for conv layer
/ convParam - param dict for conv layer
/ beta - spatial batchnorm params
/ gamma - spatial batchnorm params
/ bnParam - param dict for batchnorm
convNormReluForward:{[x;w;b;convParam;gamma;beta;bnParam]
    / convolution layer
    conv_convCache:convForwardFast[x;w;b;convParam];
    conv:conv_convCache 0;
    convCache:conv_convCache 1;

    / spatial batchnorm layer
    norm_normCache:spatialBatchNormForward[conv;gamma;beta;bnParam];
    norm:norm_normCache 0;
    normCache:norm_normCache 1;

    / final, relu layer
    out_reluCache:reluForward[norm];
    out:out_reluCache 0;
    reluCache:out_reluCache 1;

    / cache for back pass
    cache:`convCache`normCache`reluCache!(convCache;normCache;reluCache);
    (out;cache)
 };

/ backward pass for conv-batchnorm-relu convenience layer
convNormReluBackward:{[dout;cache]
    / backwards relu layer
    drelu:reluBackward[dout;cache`reluCache];

    / backwards spatial batchnorm layer
    / will be a dict of `dx`dgamma`dbeta
    dnormDgammaDbeta:spatialBatchNormBackward[drelu;cache`normCache];

    / backwards convolution layer
    / will be a dict of `dx`dw`db
    dxDwDb:convBackwardFast[dnormDgammaDbeta`dx;cache`convCache];

    / return `dx`dw`db`dgamma`dbeta! ...
    dxDwDb,`dgamma`dbeta#dnormDgammaDbeta
 };

/ convenience layer that performs a convolution, spatial batch norm,
/ relu, then a pool
/ inputs:
/ x - input to convolution layer
/ w - weight list for conv layer
/ b - bias list for conv layer
/ convParam - param dict for conv layer
/ poolParam - param dicdt for pool layer
/ beta - spatial batchnorm params
/ gamma - spatial batchnorm params
/ bnParam - param dict for batchnorm 
convNormReluPoolForward:{[x;w;b;convParam;poolParam;gamma;beta;bnParam]
    / convolution layer
    conv_convCache:convForwardFast[x;w;b;convParam];
    conv:conv_convCache 0;
    convCache:conv_convCache 1;

    / spatial batchnorm layer
    norm_normCache:spatialBatchNormForward[conv;gamma;beta;bnParam];
    norm:norm_normCache 0;
    normCache:norm_normCache 1;

    / relu layer
    relu_reluCache:reluForward[norm];
    relu:relu_reluCache 0;
    reluCache:relu_reluCache 1;
  
    / pool layer
    out_poolCache:maxPoolForwardFast[relu;poolParam];
    out:out_poolCache 0;
    poolCache:out_poolCache 1;

    / cache for backward pass
    cache:`convCache`normCache`reluCache`poolCache!(convCache;normCache;reluCache;poolCache);
    (out;cache)
 };

/ backward pass for conv-batchnorm-relu-pool convenience layer
convNormReluPoolBackward:{[dout;cache]
    / backwards pool layer
    dpool:maxPoolBackwardFast[dout;cache`poolCache];

    / backwards relu
    drelu:reluBackward[dpool;cache`reluCache];

    / backwards batchnorm
    / will be a dict of `dx`dgamma`dbeta
    dnormDgammaDbeta:spatialBatchNormBackward[drelu;cache`normCache];

    / backwards conv layer
    / will be a dict of `dx`dw`db
    dxDwDb:convBackwardFast[dnormDgammaDbeta`dx;cache`convCache];

    / return `dx`dw`db`dgamma`dbeta! ...
    dxDwDb,`dgamma`dbeta#dnormDgammaDbeta
 };










