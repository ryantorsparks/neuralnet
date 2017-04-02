system"l neural_net.q"

lg "First, we run a toy model with small data sets, 
    for this, we've pre generated the exact same random
    data this used in the python example, so load it"
{load ` sv `:assignmentInputs,x} each `toyY`toyInputDict`toyCorrectScores`toyRandomInds`toyCorrectScores

lg "compute scores"
toyScores:twoLayerNet `y _ toyInputDict

lg "scores are"
lg toyScores

toyScoreDiff:2 sum/abs toyScores-toyCorrectScores
lg "Difference between scores and correct scores are"
lg toyScoreDiff

lg "forward pass, compute loss and gradient"
toyLossGrad: twoLayerNet toyInputDict
lg toyLossGrad 0
correctToyLoss:1.30378789133
lossDiff:2 sum/abs toyLossGrad[0]-correctToyLoss

lg "Difference between your loss and correct loss:"
lg lossDiff

lg "Backward pass on toy model"

maxRelativeGradErrors:{[analyticRes;param]relError[analyticRes[param];numericalGradient[(first twoLayerNet@);toyInputDict;param]]}[toyLossGrad 1] each `w1`w2
lg "max relative gradient errors for w1 and w2 are"
lg maxRelativeGradErrors

lg "train the toy model using the same \"random\" batch indices as python uses"
toyTrained:{[d;inds]sgd[@[d;`sampleIndices;:;inds]]}/[@[toyInputDict;`reg;:;1e-5];toyRandomInds]
lg "final traing loss: ",string last toyTrained`loss
delete toyTrained from `.;
{delete x from `.}each `toyTrained`toyLossGrad`toyScores`toyScoreDiff;
.Q.gc[]


lg "Now CIFAR_10 data is already loaded - so we train our network
    To train our network we will use SGD with momentum.
    In addition, we will adjust the learning rate with an exponential 
    learning rate schedule as optimization proceeds; after each epoch, 
    we will reduce the learning rate by multiplying it by a decay rate."
lg "run 1000 iterations with starting parameters:"
lg trainStartDict:`inputTrain`outputTrain`nHidden`nClass`reg`learnRate`learnRateDecay`std`batchSize!(xTrain;yTrain;50;10;0.5;1e-4;0.95;1e-4;200)
//trainRes:1000 sgd/trainStartDict


