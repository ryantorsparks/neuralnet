/ from http://cs231n.github.io/convolutional-networks/
\l nn_util.q

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

/ zeropad in n dimensions
/ e.g. zeroPad[2 3 4 5#1f;2]
zeroPad:{[x;pad]
    shapex:shape x;
    padf:{y,(til x),y}[;pad#0N];
    c:2<count shapex;
    newShape:(c#shapex),(2*pad)+c _ shape x;
    cntList:$[c;enlist til shapex 0;()],padf each c _ shape x;
    inds:{raze y+/:x*sum not null y}/[cntList];
    newShape#0^razeo[x]@inds
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
    cache:(x;w;b;convParam);
    (out;cache)
 };

/ convolution function, backward pass
/ @param dout - upstream derivatives
/ @param cache - (x;w;b;convParam), second output of convForwardNaive
/ @return `dx`dw`db!(dx;dw;db) gradients with respect to x, w, b
convBackwardNaive:{[dout;cache]
    x:cache 0;
    w:cache 1;
    b:cache 2;
    convParam:cache 3;
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
                        tempd::d;tempinds::inds;
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
   cache:(x;poolParam);
   (out;cache)
 };


























