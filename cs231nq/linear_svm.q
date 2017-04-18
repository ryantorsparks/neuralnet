\l nn_util.q

svmLossVectorized:{[w;x;y;delta]
    scores:dot[x;w];
    correctScores:scores@'y;
    margins:0|scores-correctScores-delta;
    loss:(.5*reg*r$r:raze w)+sum/[margins]%count x;
    xmask:"f"$margins>0;
    xmask:@'[xmask;y;:;neg sum each xmask];
    dw:(reg*w)+(dot[flip[x];xmask])%count x;
    (loss;dw)
  };

svmLoss:{[d]
    x:d`x;
    y:d`y;
    N:first shape x;
    correctClassScores:x@'y;
    margins:0|1.+x-correctClassScores;
    margins:@'[margins;y;:;0.];
    loss:sum/[margins]%N;
    numPos:sum each margins>0;
    dx:"f"$margins>0;
    dx:@'[dx;y;:;neg sum each dx]%N;
    (loss;dx)
 };
