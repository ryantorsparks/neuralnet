\l nn_util.q

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
