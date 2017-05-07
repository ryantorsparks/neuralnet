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
lg "input mean is "

testDropoutP:{[x;p]
    lg "Running tests with p=",string p;
    out:first dropoutForward[x;`mode`p!`train,p];
    lg "input mean is "




