\l nn_util.q
\l fullyConnected.q
\l numerical_gradient.q
\l softmax.q
\l linear_svm.q

b:"f"$get `:assignmentInputs/fullyConnected_b
w:"f"$get `:assignmentInputs/fullyConnected_w
x:"f"$get `:assignmentInputs/fullyConnected_x
correctOut:get `:assignmentInputs/fullyConnected_correctOut

lg "Affine forwards"
out:first affineForward `x`w`b!(x;w;b)
relError[out;correctOut]

lg "Affine backwards"
b:"f"$get `:assignmentInputs/fullyConnected_bAffineBackward
w:"f"$get `:assignmentInputs/fullyConnected_wAffineBackward
x:"f"$get `:assignmentInputs/fullyConnected_xAffineBackward
dout:"f"$get `:assignmentInputs/fullyConnected_doutAffineBackward
cache:last affineForward `x`w`b!(x;w;b)
dxDwDb:affineBackward[dout;cache]
dxDwDb_num:numericalGradientArray[(first affineForward@);`x`w`b!(x;w;b);dout;]each `x`w`b
relError'[dxDwDb;dxDwDb_num]

lg "ReLU layer forward"
x:3 4#linSpace[-0.5;.5;12]
out:0|x
relError[get`:assignmentInputs/fullyConnected_correctOutReluForward;0|x]

lg "ReLU backward"
x:"f"$get`:assignmentInputs/fullyConnected_xReluBackward
dout:"f"$get`:assignmentInputs/fullyConnected_doutReluBackward
cache: last reluForward[x]
dx:reluBackward[dout;cache]
dx_num:numericalGradientArray[(first reluForward@);x;dout;`x]
relError[dx;dx_num]

lg "Sandwich layers"
w:"f"$get`:assignmentInputs/fullyConnected_wSandwich
b:"f"$get`:assignmentInputs/fullyConnected_bSandwich
x:"f"$get`:assignmentInputs/fullyConnected_xSandwich
dout:"f"$get`:assignmentInputs/fullyConnected_doutSandwich
outCache:affineReluForward `x`w`b!(x;w;b)
dxDwDb:affineReluBackward[dout;outCache 1]
dxDwDb_num:numericalGradientArray[(first affineReluForward@);`x`w`b!(x;w;b);dout;]each `x`w`b
relError'[value dxDwDb;dxDwDb_num]

lg "Loss layeres: Softmax and SVM"
y:get`:assignmentInputs/fullyConnected_yLossLayers
x:"f"$get`:assignmentInputs/fullyConnected_xLossLayers
dxNum:numericalGradient[(first svmLoss@);`x`y!(x;y);`x] 
lossDx:svmLoss `x`y!(x;y)
relError[dxNum;lossDx 1]
dxNum:numericalGradient[(first softmaxLoss@);`x`y!(x;y);`x] 
lossDx:softmaxLoss `x`y!(x;y)  
lg"loss is ",.Q.s lossDx 0
lg"relative error is ",.Q.s relError[dxNum;lossDx 1]


lg "##### Two layer network #####\n"

X:"f"$get `:assignmentInputs/fullyConnected_XTwoLayer
y:get`:assignmentInputs/fullyConnected_yTwoLayer
`. upsert `N`D`H`C`std!3,5,50,7,0.01;

lg "Testing initialization"
d:twoLayerNet.params `dimInput`dimHidden`nClass`wScale!(D;H;C;std)
wStd1:abs adev[d`w1]-std
lg "wStd1 is ",.Q.s wStd1
b1:d`b1
wStd2:abs adev[d`w2]-std
b2:d`b2
if[wStd1>0.1*std;lg"WARN: first layer weights do not seem right"]
if[not all b1=0;lg"WARN: first layer biases do not seem right"]
if[wStd2>0.1*std;lg"WARN: second layer weights do not seem right"]
if[not all b2=0;lg"WARN: second layer biases do not seem right"]

d[`w1]:(D;H)#linSpace[-.7;.3;D*H]
d[`b1]:linSpace[-.1;.9;H]
d[`w2]:(H;C)#linSpace[-.3;.4;H*C]
d[`b2]:linSpace[-.9;.1;C]
d[`x]:flip (D;N)#linSpace[-5.5;4.5;N*D]
scores:twoLayerNet.loss d
lg"scores are ",.Q.s scores
correctScores:get `:assignmentInputs/fullyConnected_correctScoresTwoLayer
scoresDiff:abs asum scores-correctScores
if[not scoresDiff<1e-6;lg"WARN: problem with test time forward pass"];

lg"Testing training loss (no regularization)"
y:0 5 1
d[`y]:y
lossGrads:twoLayerNet.loss d
correctLoss:3.4702243556
if[1e-10<abs lossGrads[0]-correctLoss;lg"WARN: problem with training time loss"];

d[`reg]:1.0
lossGrads:twoLayerNet.loss d
correctLoss:26.5948426952
if[1e-10<abs lossGrads[0]-correctLoss;lg"WARN: problem with regularization loss"];
compareNumericalGradients[d] each 0.0 0.7;

lg "###### Solver ######\n"
d:nulld;
d[`model]:`twoLayerNet








