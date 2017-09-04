/ ########## softmax functions ############

/ softmax loss in kdb
/ reg:0.1
/ w:3 4#-2 4 1 3 -2 -4 5 -3 4 5 -4 -4.
/ x:5 3#20 17 18 15 14 20 20 20 13 11 17 16 17 10 16.  
/ y:0 2 1 1 0
/ b:count[w 0]#0
softmaxLossVectorized:{[w;x;y;b;reg;step]
    scores:dot[x;w]+\:b;
    probs:expScores%sum each expScores:exp scores;
    dataLoss:sum neg[log probs@'y]%count x;
    regLoss:.5*reg*r$r:raze w;
    loss:dataLoss+regLoss;

    dscores:@'[probs;y;-;1]%count x;
    dw:dot[flip x;dscores]+reg*w;
    db:sum dscores;
    (loss;w-step*dw;b-step*db)
 };

softmaxLoss:{[d]
    x:d`x;
    y:d`y;
    probs:expScores%sum each expScores:exp x- max each x;
    N:count x;
    loss:sum neg[log probs@'y]%N;
    dx:@'[probs;y;-;1]%N;
    (loss;dx)
 };

temporalSoftmaxLoss:{[x;y;mask]
    shapeX:shape x;
    N:shapeX 0;
    T:shapeX 1;
    V:shapeX 2;
    
    xFlat:reshapeM[x;(N*T;V)];
    yFlat:reshapeM[y;N*T];
    maskFlat:reshapeM[mask;N*T];

    probs:exp xFlat-max each xFlat;
    probs%:sum each probs;
    loss:neg sumo[maskFlat*log[probs @'yFlat]i]%N;
    dxFlat:@'[probs;yFlat;-;1]%N;
    dxFlat*:maskFlat;

    dx:reshapeM[dxFlat;(N;T;V)];
    (loss;dx)
 }; 

