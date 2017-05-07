\p 5000
\l nn_util.q
\l fullyConnected.q
\l numerical_gradient.q
\l softmax.q
\l linear_svm.q
\l batchNorm.q
\l dropout.q
cifarMode:`unflattened
/\l load_cifar_data.q

lg "##############################
    Neural nets with dropout
    ##############################"


lg "##############################
    Dropout forward pass
    ##############################"

lg "test dropout forward pass"
x:randArray[500;500]+10
lg "input mean is ",string avg razeo x;

testDropoutP:{[x;p]
    lg "Running tests with p=",string p;
    outTrain:dropoutForward[x;`mode`p!`train,p] 0;
    outTest:dropoutForward[x;`mode`p!`test,p] 0;
    lg "train/test avgs are ",-3!(avg razeo@)each (outTrain;outTest);
    lg "fraction of train/test ouput set to 0 is ",-3!(avg razeo 0=)each(outTrain;outTest);
 };

testDropoutP[x;]each 0.3 0.6 0.75;


lg "##############################
    Dropout backward pass
    ##############################"

lg "run dropout backward pass and then compare to numerical gradients"
x:randArray[10;10]+10
dout:randArray . shape x
dropoutParam:`mode`p`seed!(`train;0.8;123)
outCache:dropoutForward[x;dropoutParam]
out:outCache 0;
cache:outCache 1;
dx:dropoutBackward[dout;cache]
dxNum:numericalGradientArray[(first dropoutForward[;dropoutParam]@);x;dout;`x] 
lg "dx relative error is ",string relError[dx;dxNum]

lg "##############################
    Fully connected nets with dropout
    ##############################"

lg "modify our implementation of fullyConnectedNet to use dropout, 
    specifically if the net gets a non zero `dropout param, then the net
    should add dropout immediately after ever ReLU nonlinearity. Then we
    numerically check our implementation"

`. upsert `N`D`H1`H2`C!2 15 20 30 10;
x:randArray[N;D]
y:N?C
startd:(!). flip ((`dimHidden;H1,H2);(`dimInput;D);(`nClass;C);(`wScale;5e-2);(`seed;123);(`x;x);(`y;y));

lossGradCheckOneDropout:{[startd;dropout]
    lg "running check with dropout = ",string dropout;
    initd:fullyConnectedNet.init @[startd;`dropout;:;dropout];
    lossGrad:fullyConnectedNet.loss initd;

    lg "initial loss is ",string lossGrad 0;
    gradCheckDict:@[((raze key[startd],initd[`wParams`bParams]),`useBatchNorm`wParams`bParams`dropout)#initd;`model;:;`fullyConnectedNet];

    lg "comparing numerical gradients, reg=0";
    compareNumericalGradients[gradCheckDict;0]
 };

lossGradCheckOneDropout[startd;]each 0 0.25 0.5;

lg "##############################
    Regularization experiment
    ##############################"

lg "we will now train a pair of two layer networks on 500 training
    examples:
      1. with no dropout
      2. with dropout propability = 0.75
    then we will plot results"


numTrain:500
smallData:`xTrain`yTrain`xVal`yVal!(numTrain#xTrain;numTrain#yTrain;xVal;yVal)
startd:smallData,(!). flip (`model`fullyConnectedNet;(`dimHidden;1#500);(`nClass;10);(`wScale;2e-2);(`numEpochs;25);(`batchSize;100);(`updateRule;`adam);(`optimConfig;enlist[`learnRate]!enlist 5e-4);(`printEvery;100));

trainOneDropout:{[startd;dropout]
    lg "running training for dropout = ",string dropout;
    res:solver.train @[startd;`dropout;:;dropout];
 };

res:trainOneDropout[startd;]each 0.0 0.75
plotRes:update iteration:til count i by method from ungroup (`method,k)!/:`noDropout`dropout,'res@\:k:`trainAccHistory`valAccHistory

/ define pivot lambda, strings wScale for the line chart in qstudio
/ just input either `trainAccHistory`valAccHistory
piv:{[res;pivCol]?[res;();{x!x}1#`iteration;(#;enlist `noDropout`dropout;(!;`method;pivCol))]}[plotRes;]

lg "use a scatter plot to chart:
    piv `trainAccHistory
    piv `valAccHistory"


