\l nn_util.q
\l fullyConnected.q
\l numerical_gradient.q
\l softmax.q
\l linear_svm.q
cifarMode:`unflattened
\l load_cifar_data.q

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
d:twoLayerNet.init `dimInput`dimHidden`nClass`wScale!(D;H;C;std)
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
scoresDiff:abs sumo scores-correctScores
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
/ key d has `dimInput`dimHidden`nClass`wScale`reg`b1`w1`b2`w2`x`y
compareNumericalGradients[d] each 0.0 0.7;

lg "\n###### Solver ######\n"
startd:`model`xTrain`yTrain`xVal`yVal`updateRule`optimConfig`learnRateDecay`numEpochs`batchSize`printEvery!(`twoLayerNet;xTrain;yTrain;xVal;yVal;`sgd;enlist[`learnRate]!enlist 1e-3;0.95;9;200;100)
lg "run training, should be able to achieve > 50% validation accuracy"
/
res: solver.train startd;
lg "plot loss history, validation and training accuracy in an IDE e.g qstudio using scatterplots:"
lg"loss history: ([]iteration:til count res`lossHistory;loss:res`lossHistory)"
lg"train history: ([]epoch:til 1+ res`numEpochs;loss: res`trainAccHistory)"
lg"validation history: ([]epoch:til 1+ res`numEpochs;loss: res`valAccHistory)"
\

lg "######## Multi layer network ###########"
`. upsert `N`D`H1`H2`C!2 15 20 30 10
d:`dimHidden`dimInput`nClass`reg`wScale!(H1,H2;D;C;0.0;5e-2)  
x:randArray . N,D
y:N?C
startd: d,`x`y`reg!(x;y;0.0)
initd:fullyConnectedNet.init startd
res: fullyConnectedNet.loss fullyConnectedNet.init startd
lg "compare numerical gradients for reg in 0.0 3.14"
gradCheckDict:@[((raze key[startd],initd[`wParams`bParams]),`wParams`bParams`layerInds`dropoutParam)#initd;`model;:;`fullyConnectedNet]
compareNumericalGradients[gradCheckDict]each 0.0 3.14;

lg "Overfit a small data set using a 3-layer net"
numTrain:50
smallData:`xTrain`yTrain`xVal`yVal!(numTrain#xTrain;numTrain#yTrain;xVal;yVal)
startd:smallData,`model`dimHidden`nClass`reg`learnRate`learnRateDecay`wScale`updateRule`optimConfig`numEpochs`batchSize`printEvery!(`fullyConnectedNet;100 100;10;0.0;0.01;0.95;0.01;`sgd;enlist[`learnRate]!enlist 0.01;20;25;10)
res: solver.train startd

lg "Use a 5 layer net to overfit 50 training examples"
numTrain:50
smallData:`xTrain`yTrain`xVal`yVal!(numTrain#xTrain;numTrain#yTrain;xVal;yVal)
startd:smallData,`model`dimHidden`nClass`reg`learnRate`learnRateDecay`wScale`updateRule`optimConfig`numEpochs`batchSize`printEvery!(`fullyConnectedNet;4#100;10;0.0;0.021;0.95;0.036;`sgd;enlist[`learnRate]!enlist 0.021;20;25;10)
res:solver.train startd


lg "\n########### SGD and momentum ############"
N:4
D:5
w:(N,D)#linSpace[-0.4;0.6;N*D] 
dw:(N,D)#linSpace[-0.6;0.4;N*D] 
v:(N;D)#linSpace[0.6;0.9;N*D]
config:`learnRate`velocity!(0.001;v)
nextWConfig:sgdMomentum[w;dw;config]
lg"read in expected nextW and velocity"
relError[nextWConfig 0;get`:assignmentInputs/fullyConnected_expectedNextWSgdMomentum]
relError[nextWConfig[1;`velocity];get `:assignmentInputs/fullyConnected_expectedVelocitySgdMomentum]

lg"train a 6 layer network with sgd and sgdMomentum, sgdMomentum should converge faster"
numTrain:4000
smallData:`xTrain`yTrain`xVal`yVal!(numTrain#xTrain;numTrain#yTrain;xVal;yVal)
startd:smallData,(!) . flip (`model`fullyConnectedNet;(`dimHidden;5#100);(`nClass;10);(`learnRate;0.01);(`wScale;5e-2);(`optimConfig;(enlist `learnRate)!enlist 0.01);(`numEpochs;5);(`batchSize;100))
lg "running training with sgd"
resSgd:solver.train @[startd;`updateRule;:;`sgd] 
lg "running training with sgdMomentum"
resSgdMomentum:solver.train @[startd;`updateRule;:;`sgdMomentum] 
lg "plot loss histories for each, e.g. in qstudio"
lg "scatter plot of: ([]iteration:til count resSgd`lossHistory;lossSgd:resSgd`lossHistory;lossSgdMomentum:resSgdMomentum`lossHistory)"
lg "line chart of: ([]iteration:string til 1+count resSgd`valAccHistory;trainAccSgd:0.,resSgd`valAccHistory;lossSgdMomentum:0.,resSgdMomentum`trainAccHistory)"   
lg "line chart of: ([]iteration:string til 1+count resSgd`valAccHistory;valAccSgd:0.,resSgd`valAccHistory;lossSgdMomentum:0.,resSgdMomentum`valAccHistory)"

lg "\n############ rms prop and adam update ###########\n"
lg "test rmsProp implementatoin, error should be less than 1e-7"
N:4
D:5
w:(N,D)#linSpace[-0.4;0.6;N*D]
dw:(N,D)#linSpace[-0.6;0.4;N*D]
cache:(N,D)#linSpace[0.6;0.9;N*D]
config:`learnRate`cache!(1e-2;cache)
nextWConfig:rmsProp[w;dw;config]
lg "compare nextW and cache to expected results"
relError[nextWConfig 0;get `:assignmentInputs/fullyConnected_expectedNextWRmsProp]
relError[nextWConfig[1]`cache;get `:assignmentInputs/fullyConnected_expectedCacheRmsProp]

lg "test adam implementation, error shoudl be ~ 1e-7 or less"
N:4
D:5
w:(N,D)#linSpace[-0.4;0.6;N*D]
dw:(N,D)#linSpace[-0.6;0.4;N*D]
mAdam:(N,D)#linSpace[0.6;0.9;N*D]
vAdam:(N,D)#linSpace[0.7;0.5;N*D]
config:`learnRate`mAdam`vAdam`tAdam!(0.01;mAdam;vAdam;5)
nextWConfig:adam[w;dw;config]
lg "compare nextW, v and m to expected values"
relError[nextWConfig 0;get`:assignmentInputs/fullyConnected_expectedNextWAdam]
relError[nextWConfig[1;`vAdam];get`:assignmentInputs/fullyConnected_expectedVAdam]
relError[nextWConfig[1;`mAdam];get`:assignmentInputs/fullyConnected_expectedMAdam]


