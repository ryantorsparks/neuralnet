\p 5001
\l load_all.q
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
    Convolution: max pool naive forward
    ##############################"

lg "we now implement a naive forward pass for max pool layer"

xShape:2 3 4 4
x:xShape#linSpace[-0.3;0.4;prd xShape]
poolParam:`poolWidth`poolHeight`stride!2 2 2
outCache:maxPoolForwardNaive[x;poolParam]

lg "check relative error from expected output"
relError[outCache 0;pget[`convNets;`expectedForwardMaxPool]] 

lg "##############################
    Convolution: max pool naive backward
    ##############################"

lg "we now implement the naive backward pass for max pool layer"

x: rad 3 2 8 8 
dout:rad 3 2 4 4 
poolParam:`poolHeight`poolWidth`stride!2 2 2
outCache:maxPoolForwardNaive[x;poolParam]
dx:maxPoolBackwardNaive[dout;outCache 1]
dxNum:numericalGradientArray[(first maxPoolForwardNaive[;poolParam]@);x;dout;`x]

lg "check relative error against numerical gradient"
relError[dx;dxNum]

lg "##############################
    Convolutional 'sandwich' layers
    ##############################"

lg "##############################
    Testing conv relu pool forward
    ##############################"

x:rad 2 3 16 16
w:rad 3 3 3 3 
b:rad 3
dout:rad 2 3 8 8
convParam:`stride`pad!1 1
poolParam:`poolHeight`poolWidth`stride!2 2 2

outCache:convReluPoolForward[x;w;b;convParam;poolParam]
out:outCache 0;cache:outCache 1;
dxDwDb:convReluPoolBackward[dout;cache]

lg "now get numerical grads for comparison"
dxNum:numericalGradientArray[(first convReluPoolForward[;w;b;convParam;poolParam]@);x;dout;`x]
dwNum:numericalGradientArray[(first convReluPoolForward[x;;b;convParam;poolParam]@);w;dout;`w]
dbNum:numericalGradientArray[(first convReluPoolForward[x;w;;convParam;poolParam]@);b;dout;`b]
lg "check relative error against numerical gradient"
relError'[value dxDwDb;(dxNum;dwNum;dbNum)]

lg "##############################
    Testing conv relu forward
    ##############################"

x:rad 2 3 8 8
w:rad 3 3 3 3
b:rad 3
dout:rad 2 3 8 8
convParam:`stride`pad!1 1

outCache:convReluForward[x;w;b;convParam]
cache:outCache 1
dxDwDb:convReluBackward[dout;cache]

lg "now get numerical grads for comparison"
dxNum:numericalGradientArray[(first convReluForward[;w;b;convParam]@);x;dout;`x]
dwNum:numericalGradientArray[(first convReluForward[x;;b;convParam]@);w;dout;`w]
dbNum:numericalGradientArray[(first convReluForward[x;w;;convParam]@);b;dout;`b]

lg "check relative error against numerical gradient"
relError'[value dxDwDb;(dxNum;dwNum;dbNum)]

lg "##############################
    Three layer convnet
    ##############################"

x:rad 50 3 32 32
y:50?10
initd:threeLayerConvNet.init `x`y!(x;y)
lossGrad:threeLayerConvNet.loss initd
lg "initial loss (no regularization):"
lossGrad 0
lossGrad2:threeLayerConvNet.loss @[initd;`reg;:;0.5]
lg "initial loss (with regularization):"
lossGrad2 0
