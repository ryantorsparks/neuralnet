\l nn_util.q
\l numerical_gradient.q

/ relative errors
relError:{[x;y]max/[abs[x-y]%1e-8|sum abs(x;y)]}

/ like np.random.randn
wInit:{(x;y)#sqrt[-2*log n?1.]*cos[2*3.14159265359*(n:x*y)?1.]}

/ just to make sure we don't forget keys in our input dicts
checkInputs:{[d;specials]
    required:`x`w1`w2`b1`b2,specials;
    if[`y in k:key d;required,:`y`reg];
    if[not all `w1`w2`b1`b2 in k;required,:`nHidden`batchSize];
    if[count missing:required except k;'"missing the following keys from input dict: ",-3!missing];
 };

/ relu second layer, softmax output layer 
twoLayerNet:{[d]
    checkInputs[d;()];
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

    / backwards
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

/ stochastic gradient descent
/ e.g.
/ res:{[d;inds]sgd[@[d;`sampleIndices;:;inds]]}/[get`:assignmentData/inputd;get`:assignmentData/randomInds]  
/ res:sgd/[1000;d:`inputTrain`outputTrain`nHidden`nClass`reg`learnRate`learnRateDecay`std`batchSize!(xTrain;yTrain;50;10;0.5;1e-4;0.95;1e-4;200)]
sgd:{[d]
    if[not `numTrain in key d;d[`numTrain]:count d`inputTrain];
    if[not `numFeatures in key d;d[`numFeatures]:count d[`inputTrain] 0];
    if[not `cnt in key d;d[`cnt]:0];
    if[not `accuracy in key d;d[`accuracy]:enlist 0.];
    if[not `loss in key d;d[`loss]:enlist 0n];
    numTrain:d`numTrain;
    numFeatures:d`numFeatures;
    batchSize:$[`batchSize in key d;d`batchSize;numTrain];
    sampleIndices:$[`sampleIndices in key d;
                      d`sampleIndices;
                      neg[batchSize]?numTrain
                  ];
    / allow for running sgd with no input weights/biases
    std:$[`std in key d;d`std;1.0];
    if[not`w1 in key d;d[`w1]:std*wInit[numFeatures;d`nHidden]];
    if[not`w2 in key d;d[`w2]:std*wInit[d`nHidden;d`nClass]];
    if[not`b1 in key d;d[`b1]:d[`nHidden]#0f];
    if[not`b2 in key d;d[`b2]:d[`nClass]#0f];
    d[`x`y]:d[`inputTrain`outputTrain]@\:sampleIndices;
    checkInputs[d;`learnRate];
    
    lossGrad:twoLayerNet[d];
    d[`loss],:lossGrad 0;
    grad:lossGrad 1;
    d[vars]-:abs[d`learnRate]*grad vars:`w1`b1`w2`b2;
    d[`accuracy],:  accuracy: avg predict[`x`w1`w2`b1`b2#d]=d`y;
    if[0=d[`cnt] mod 100;
        lg "TRAINING PROGRESS: cnt, learnRate, loss, accuracy are ",-3!(d`cnt;d`learnRate;lossGrad 0;accuracy)
    ];
    if[(0=d[`cnt]mod numTrain%batchSize)and `learnRateDecay in key d;d[`learnRate]*:d`learnRateDecay];
    d[`cnt]+:1;
    d
 };

/ optimize hyperparameters
/ vary learning rate and regularization parameters to try and find
/ better paramters for learning
/ @global xVal (input validation data)
/ @global yVal (output validation data)
/ @global xTrain (input training data - too big for 32 bit kdb to handle sending across)
/ @global yTarin (input output data)
varyHyperParams:{[startDict;iterations;numRandoms;lrRange;regRange]
    randomLearnRates:lrRange[0]+numRandoms?lrRange[1]-lrRange 0;
    randomRegs:regRange[0]+numRandoms?regRange[1]-regRange 0;
    lg "running varyHyperParams with: \n",.Q.s(randomLearnRates;randomRegs);
    inputs:flip (randomLearnRates;randomRegs);
    lg "running varyHyperParams with: \n",.Q.s inputs;
    ( {[d;n;lr;reg]
          lg"changing learnRate and reg, adding xTrain and yTrain (mem issues)";
          d[`learnRate`reg`inputTrain`outputTrain]:(lr;reg;xTrain;yTrain);
          lgToken:" lr = ",string[lr],", reg = ",string reg;
          lg "running ",string[n]," iterations of sgd for ",lgToken;
          res:n sgd/d;
          valAccuracy: avg yVal=predict `x`w1`w2`b1`b2!enlist[xVal],res`w1`w2`b1`b2;
          lg "validation accuracy for ",lgToken," is ",string valAccuracy;
          (lr;reg;valAccuracy;res`w1`w2`b1`b2)
      }[startDict;iterations;;] . 
    ) peach inputs
 };

/ predict function, determines which class
/ inputs:
/ d: dictionary, using following names:
/    where C= numClass, H=hidden layer num
/    and D is dimension of inputs X (count x 0)
/      w1: weights (D H)
/      w2: weights (H C)
/      b1: biases (list length H)
/      b2: biases (list length C)
/      x: shape (N D), N, D-dimensional data points to classify 
/ returns:
/ list of length N, predicted labels of elements of x (should
/   be between 0 and C inclusive, e.g digits will be from 0 to 9).
/ e.g:
/    res:sgd/[1000;d:`inputTrain`outputTrain`nHidden`nClass`reg`learnRate`learnRateDecay`std`batchSize!(xTrain;yTrain;50;10;0.5;1e-4;0.95;1e-4;200)]
/    predict `x`w1`w2`b1`b2!enlist[xVal],res`w1`w2`b1`b2;
predict:{[d]
    a1:0|z1:dot[d`x;d`w1]+\:d`b1;
    scores:dot[a1;d`w2]+\:d`b2;
    (first idesc@)each scores
 };
