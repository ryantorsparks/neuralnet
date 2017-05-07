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
    Regularization experiment
    ##############################"


