/ from http://cs231n.github.io/convolutional-networks/
\l nn_util.q

/ pad matrix m with n zeros
zeroPad2D:{[n;m]p,((nz,/:m),\:nz:n#0.),p:n#enlist (count[first m]+2*n)#0.} 

/ indice generating func (used for sriding across a matrix)
strideInds:{[vSize;vDepth;fSize;stride;strides]
    yAxis:raze (raze til[fSize]+/:vSize*til fSize)+/:vSize*vSize*til vDepth;
    xAxis:(raze (stride*til strides)+/:(vSize*stride)*til strides);
    xAxis+/:yAxis
 };

poolInds:{[mSize;fSize;stride;strides]
    {x cross x} enlist each til[fSize]+/:stride*til strides
 };

/ convolution function
/ @param V - Volume (matrix, not padded yet, should be square)
/ @param F - Filter (matrix, should be square)
/ @param b - bias (list of floats, should be same as first[shape F])
/ @param padSize - number of zeros to pad volume matrix V with 
/ @param stride - the stride length across padded V
/ e.g.
/ V:((1 1 1 1 0f;0 1 2 1 1f;0 0 0 2 0f;1 2 0 0 2f;0 2 0 1 1f);(1 0 2 0 0f;0 2 2 2 0f;0 1 2 1 0f;1 1 0 0 1f;2 0 2 1 2f);(1 0 0 1 0f;2 1 1 2 0f;2 0 0 2 0f;2 0 0 0 0f;1 0 0 1 1f))
/ F:(((0 1 0f;1 1 1f;0 -1 0f);(-1 0 -1f;-1 -1 1f;1 1 1f);(1 -1 0f;1 1 1f;0 -1 -1f));((0 -1 1f;1 1 0f;0 -1 0f);(1 0 1f;0 -1 0f;-1 1 1f);(1 0 1f;-1 1 0f;0 0 1f)))
/ b:0 1f
/ conv[V;F;b;1;2]
conv:{[V;F;b;padSize;stride]

    / pad volume V
    paddedV:zeroPad[V;padSize];

    / size of padded V
    paddedVSize:last shape paddedV;

    / filter size, F should have shape (#of filters;n;n;depth)
    fSize:shape[F] 2;

    / flatten filters
    flatF:razeo each F;

    / output dimension/total number of strides
    strides:1+(paddedVSize-fSize)div stride;

    / filter indices (used to index into volume V and create one matrix to multiply)
    filterInds:strideInds[paddedVSize;shape[paddedV]0;fSize;stride;strides];

    / perform the convolution/matrix multiplication
    (2#strides)#/:b+dot[flatF;razeo[paddedV] filterInds]
 }

/ old version, less efficient indexing/flip with filterInds
convOld:{[V;F;b;padSize;stride]

    / pad volume V
    paddedV:zeroPad[V;padSize];

    / size of padded V
    paddedVSize:last shape paddedV;

    / filter size, F should have shape (#of filters;n;n;depth)
    fSize:shape[F] 2;

    / flatten filters
    flatF:razeo each F;

    / output dimension/total number of strides
    strides:1+(paddedVSize-fSize)div stride;

    / filter indices (used to index into volume V and create one matrix to multiply)
    filterInds:poolInds[paddedVSize;fSize;stride;strides];

    / perform the convolution/matrix multiplication
    / TODO: redo this more efficiently
    (2#strides)#/:b+dot[flatF;flip {[m;i;j]2 raze/ .[m;(::;i;j)]}[paddedV] ./:filterInds]
 }

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

/ older, less efficient version
poolOld:{[m;fSize;stride]
    convSize:count m 0;
    strides:1+(convSize-fSize)div stride;

    / pool indices
    inds:poolInds[convSize;fSize;stride;strides];

    / run max across each sub quadrant
    (2#strides)#{[m;inds]max raze m . inds}[m]each inds
 }

convNetNaiveOld:{[x;w;b;convParam]
    stride:convParam`stride;
    pad:convParam`pad;
    xShape:shape x;
    N:xShape 0;C:xShape 1;H:xShape 2;W:xShape 3;
    wShape:shape w;
    F:wShape 0; HH:wShape 2;WW:wShape 3;
    hout:`long$1+(H+(2*pad)-HH)%stride;
    wout:`long$1+(W+(2*pad)-WW)%stride;
    out:("i"$N,F,hout,wout)#0f;

    / loop:
    / d is `F`x`hout`wout`HH`WW`stride`pad!(F;x;hout;wout;HH;WWstride;pad)
    convLoop:{[d]
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
                        (inds;d[`convB]+sumo convArea*d`convW)
                    }[d;] each inds,/:til d`wout
                }[d;] each inds,/:til d`hout
            }[d;] each ind,/:til d`F
        }[d;]each til d`N
    };
    res:3 raze/convLoop `N`F`x`w`b`hout`wout`HH`WW`stride`pad!(N;F;x;w;b;hout;wout;HH;WW;stride;pad);
    tempres::res;
    newOut:./[out;res[;0];:;res[;1]];
    cache:(x;w;b;convParam);
    (newOut;cache)
 };

convForwardNaive:{[x;w;b;convParam]
    stride:convParam`stride;
    pad:convParam`pad;
    xShape:shape x;
    N:xShape 0;C:xShape 1;H:xShape 2;W:xShape 3;
    wShape:shape w;
    F:wShape 0; HH:wShape 2;WW:wShape 3;
    hout:`long$1+(H+(2*pad)-HH)%stride;
    wout:`long$1+(W+(2*pad)-WW)%stride;
    outShape:"i"$N,F,hout,wout;

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
    res:convInner `N`F`x`w`b`hout`wout`HH`WW`stride`pad!(N;F;x;w;b;hout;wout;HH;WW;stride;pad);
    cache:(x;w;b;convParam);
    (res;cache)
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
    
zeroPadSlow1:{[x;pad]
    shapex:shape x;
    f:{y,(x#1),y}[;pad#0N];
    c:2<count shapex;
    newShape:(c#shapex),(2*pad)+c _ shape x;
    cntList:$[c;enlist first[shapex]#1;()],f each c _ shape x;
    inds:{raze x*/:\:y}/[cntList];
    newShape#0^razeo[x] @[inds;w;:;til count w:where not null inds]
 };

zeroPadSlow2:{[x;pad]
    shapex:shape x;
    f:{y,til[x],y}[;pad#0N];
    c:2<count shapex;
    newShape:(c#shapex),(2*pad)+c _ shape x;
    inds:cross/[$[c;enlist til shapex 0;()],f each c _ shape x];
    newShape#0^razeo[x] @[pfi;where not null pfi:prd flip inds;:;til prd shape x]
 };

/ older, slower version
zeroPadSlow3:{[x;pad]
    shapex:shape x;
    f:{y,til[x],y}[;pad#-1];
    c:2<count shapex;
    newShape:(c#shapex),(2*pad)+c _ shape x;
    inds:cross/[$[c;enlist til shapex 0;()],f each c _ shape x];
    newShape#0^x ./:inds
 };
