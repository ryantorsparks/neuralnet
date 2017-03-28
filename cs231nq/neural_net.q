\l nn_util.q
\l numerical_gradient.q 

/ relative errors
relError:{[x;y]max/[abs[x-y]%1e-8|sum abs(x;y)]}

checkInputs:{[d]
    required:`x`w1`w2`b1`b2;
    if[`y in k:key d;required,:`y`reg];
    if[count missing:required except k;'"missing the following keys from input dict: ",-3!missing];
 };
 
epoch:{[d]
    checkInputs[d];
    w1:d`w1;
    w2:d`w2;
    x:d`x;
    b1:d`b1;
    b2:d`b2;
    reg:d`reg;

    / forward
    z1:dot[x;w1]+\:b1;
    a1:0|z1;
    scores:dot[a1;w2]+\:b2;
    if[not `y in key d;:scores];
    y:d`y;
    expScores:exp scores;
    probs:expScores%sum each expScores;
    correctLogProbs:neg log probs@'y;
    dataLoss:avg correctLogProbs;
    regLoss:.5*d[`reg]*r$r:raze[w1],raze w2;
    loss:dataLoss+regLoss;

    dscores:@'[probs;y;-;1]%count x;
    dw2: dot[flip a1;dscores];
    db2: sum dscores;
    
    dhidden:dot[dscores;flip w2]*a1>0;
    dw1:dot[flip x;dhidden];
    db1:sum dhidden;
    
    dw2+:w2*reg;
    dw1+:w1*reg;
    (loss;`w1`w2`b1`b2!(dw1;dw2;db1;db2))
 };


/ numIters:100;
/ reg:1e-5;
/ batchSize:200;
/ learnRate:0.001;
/ learnRateDecay:0.95
train:{[d]
    inputTrain:d`inputTrain;
    outputTrain:d`outputTrain;
    inputValid:d`validTrain;
    outputValid:d`outputValid;
    learnRate:d`learnRate;
    learnRateDecay:d`learnRateDecay;
    reg:d`reg;
    numIters:d`numIters;
    batchSize:n`batchSize;
    numTrain:count inputTrain;
    iterationsPerEpoch:1|numTrain%batchSize;
    lossHistory:();
    trainAccuraccyHistory:();
    valAccuracyHistory:();
    
 }; 
    
sgd:{[d]
    if[count missing:`reg`learnRate`batchSize`inputTrain`outputTrain`nHidden except key d;
       '"missing following keys from input dict: ",-3!missing
      ];
    sampleIndices:neg[d`batchSize]?d`numTrain;
    d:d,
    (!) . flip
       ( 
         (`x;d[`inputTrain]sampleIndices);
         (`y;d[`outputTrain]sampleIndices);
         (`w1;wInit[count x 0;d`nHidden]);
         (`w2;wInit[d`nHidden;d`nClass]);
         (`b1;d[`nHidden]#0f);
         (`b2;d[`nClass]#0f)
       );
    lossGrad:epoch[d];
    d[`loss],:lossGrad 0;
    grad:lossGrad 1;
    d[vars]-:abs[d`learnRate]*grad vars:`w1`b1`w2`b2;
    d
 };

/ predict function, determines which class
/ inputs:
/ d, which are (dimensions), where C= numClass, H=hidden layer num
/ and D is dimension of inputs X (count x 0)
/ w1: weights (D H)
/ w2: weights (H C)
/ b1: biases (list length H)
/ b2: biases (list length C)
/ x: shape (N D), N, D-dimensional data points to classify 
/ returns:
/ list of length N, predicted labels of elements of x (should
/   be between 0 and C inclusive, e.g digits will be from 0 to 9).
predict:{[d]
    a1:0|z1:dot[d`x;d`w1]+d`b1;
    scores:dot[a1;d`w2]+d`b2;
    (first idesc@)each scores
 };



twoLayerNet:{[d]
    checkInputs[d];
    w1:d`w1;
    w2:d`w2;
    x:d`x;
    b1:d`b1;
    b2:d`b2;
    reg:d`reg;

    / forward
    layer1:dot[x;w1]+\:b1;
    layer2:0|layer1;
    layer3:dot[layer2;w2]+\:b2;
    scores:layer3;
    if[not `y in key d;:scores];
    expLayer3:exp layer3;
    rows:sum each expLayer3;
    layer4:avg neg[layer3@'y]+log rows;
    loss:layer4+.5*reg*r$r:2 raze/w1,w2;

    / backpass
    dlayer4:1.0;
    dlayer3:dlayer4*@'[expLayer3%rows;y;-;1f]%count x;
    dlayer2:dot[dlayer3;flip w2];
    dlayer1:dlayer2*layer1>0;
    d1w:dot[flip x;dlayer1]+w1*reg;
    w2:dot[flip layer2;dlayer3]+w2*reg;

    db1:sum dlayer1;
    db2:sum dlayer3;
    (loss;`w1`w2`b1`b2!(dw1;dw2;db1;db2))
 };
