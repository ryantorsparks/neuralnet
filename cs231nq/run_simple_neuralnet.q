if[not 0>system"s";-1"this script should be run with multi process, so start with negative s, e.g. \"q run.q -s -2\"";exit 0];
system"l load_all.q"
system"l load_cifar_data.q"

lg "##############################
    Implementing a simple neural network
    ##############################"

lg "First, we run a toy model with small data sets, 
    for this, we've pre generated the exact same random
    data this used in the python example, so load it"
{load ` sv `:assignmentInputs,x} each `toyY`toyInputDict`toyCorrectScores`toyRandomInds`toyCorrectScores
lg "##############################
    compute scores
    ##############################"

toyScores:twoLayerNet.loss `y _ toyInputDict

lg "scores are"
toyScores

toyScoreDiff:2 sum/abs toyScores-toyCorrectScores
lg "Difference between scores and correct scores are"
toyScoreDiff

lg "##############################
    forward pass, compute loss and gradient
    ##############################"

toyLossGrad: twoLayerNet.loss toyInputDict
toyLossGrad 0
correctToyLoss:1.30378789133
lossDiff:2 sum/abs toyLossGrad[0]-correctToyLoss

lg "Difference between your loss and correct loss:"
lossDiff

lg "##############################
    Backward pass on toy model
    ##############################"

maxRelativeGradErrors:{[analyticRes;param]relError[analyticRes[param];numericalGradient[(first twoLayerNet.loss@);toyInputDict;param]]}[toyLossGrad 1] each `w2`b2`w1`b1
lg "max relative gradient errors for w1 and w2 are"
maxRelativeGradErrors

lg "##############################
    train the toy model using the same \"random\" batch indices as python uses
    ##############################"

toyTrained:{[d;inds]simpleSgd[@[d;`sampleIndices;:;inds]]}/[@[toyInputDict;`reg;:;1e-5];toyRandomInds]
lg "final traing loss: ",string last toyTrained`loss

lg "##############################
    Train a network
    ##############################"

lg "Now CIFAR_10 data is already loaded - so we train our network
    To train our network we will use SGD with momentum.
    In addition, we will adjust the learning rate with an exponential 
    learning rate schedule as optimization proceeds; after each epoch, 
    we will reduce the learning rate by multiplying it by a decay rate."
lg "run 1000 iterations with starting parameters:"
lg trainStartDict:`inputTrain`outputTrain`nHidden`nClass`reg`learnRate`learnRateDecay`std`batchSize!(`float$xTrain;yTrain;50;10;0.5;1e-4;0.95;1e-4;200)
trainRes:1000 simpleSgd/trainStartDict

lg "##############################
    Debug the training
    ##############################"

lg "Check the accuracy with validation data"
valAccuracy:avg yVal=predict `x`w1`w2`b1`b2!enlist[`float$xVal],trainRes`w1`w2`b1`b2;
lg "Validation accuracy was only ",(string valAccuracy),", which isn't great.
    We can look at things like the loss and accuracy history to try and tweak the 
    hyperparamters"
lg "loss history (better plotted using an IDE/chart"
trainRes`loss
lg "accuracy history (again, better in a chart"
trainRes`accuracy

lg "We use random search to find better hyperparameters (learning 
    rate and regularization parameter), keeping reg between 0.35 and 0.6,
    and lr between 0.0005 and 0.0015, lr and regs are respectively:"
numRandoms:10
(::)randomLearnRates:0.0005+numRandoms?0.001
(::)randomRegs:0.35+numRandoms?0.25

lg "##############################
    Tune hyperparameters
    ##############################"

lg "for this test, we do multi process peach"
slaves:abs system"s"
{system"q -p ",string[18800+x]," -g 1 &"} each til slaves
system"sleep 3"

lg "load functions and cifar data into slaves"
.z.pd:`u#hopen each 18800+til slaves
.z.pd@\:"system\"l load_all.q\""
.z.pd@\:"system\"l load_cifar_data.q\""

lg "find better learning rate and regularization parameters"
res:varyHyperParams[`inputTrain`outputTrain _ trainStartDict;2000;4;0.0015 0.005;0.35 0.5];

/ example, good parameters: 0.0009866434 0.5184204
bestParams:res first idesc res[;2]
lg "Best params found to be ",(-3!bestParams 0 1),", try running on test data, test accuracy is: "
avg yTest=predict `x`w1`w2`b1`b2!enlist[`float$xTest],bestParams 3
