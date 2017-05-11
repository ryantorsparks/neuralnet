\p 5001
\l nn_util.q
\l fullyConnected.q
\l numerical_gradient.q
\l softmax.q
\l linear_svm.q
\l batchNorm.q
\l dropout.q
\l convnet.q
cifarMode:`unflattened
/\l load_cifar_data.q


lg "##############################
    Convolutional Networks
    ##############################"


lg "##############################
    Convolution: naive forward pass
    ##############################"

lg "The core of a convolutiona network is the convolution operation.
    Here we implement a naive (slow) forward pass"

xShape:2 3 4 4 
wShape:3 3 4 4 
x:xShape#linSpace[-0.1;0.5;prd xShape]
w:wShape#linSpace[-0.2;0.3;prd wShape]
b:linSpace[-0.1;0.2;3]
convParam:`stride`pad!2 1

out:first convForwardNaive[x;w;b;convParam]
lg "relative error with expected result is ",.Q.s1 relError[out;pget[`convNets;`expectedForwardOut]]

lg "##############################
    Convolution: naive backward pass
    ##############################"

lg "now we implement the backward pass for the convolution operation,
    then we do a numerical gradient check"

x:rad 4 3 5 5
w:rad 2 3 3 3
b:rad 2
dout:rad 4 2 5 5
convParam:`stride`pad!1 1

lg "first get numerical grads"
dxNum:numericalGradientArray[(first convForwardNaive[;w;b;convParam]@);x;dout;`x]
dwNum:numericalGradientArray[(first convForwardNaive[x;;b;convParam]@);w;dout;`w]
dbNum:numericalGradientArray[(first convForwardNaive[x;w;;convParam]@);b;dout;`b]

lg "now run batchNorm forward then backward"
outCache:convForwardNaive[x;w;b;convParam]
dxDwDb:convBackwardNaive[dout;outCache 1]

lg "numerical errors for `dx`dw`db are: "
relError'[value dxDwDb;(dxNum;dwNum;dbNum)]

lg "##############################
    Convolution: max pool layer
    ##############################"

lg "we now implement a naive forward pass for max pool layer"

xShape:2 3 4 4
x:xShape#linSpace[-0.3;0.4;prd xShape]
poolParam:`poolWidth`poolHeight`stride!2 2 2
outCache:maxPoolForwardNaive[x;poolParam]

lg "check relative error from expected output"
relError[outCache 0;pget[`convNets;`expectedForwardMaxPool]] 
