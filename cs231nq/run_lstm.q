@[system;"p 5000";{-1"WARNING: failed to set port to 5000";}]
\l load_all.q
\l load_coco_data.q
/ set runAll to 1b only if you want to run everything
runAll:0b

lg "##############################
    Image captioning with LSTMs
    ##############################"

lg "##############################
    LSTM step forward
    ##############################"

`N`D`H set'3 4 5
x:(N;D)#linSpace[-0.4;1.2;N*D]
prevH:(N;H)#linSpace[-0.3;0.7;N*H]
prevC:(N;H)#linSpace[-0.4;0.9;N*H]
wx:(D,4*H)#linSpace[-2.1;1.3;D*4*H]
wh:(H*1 4)#linSpace[-0.7;2.2;H*4*H]
b:linSpace[0.3;0.7;4*H]
res:lstmStepForward `x`prevH`prevC`wx`wh`b!(x;prevH;prevC;wx;wh;b)

expectedC:(0.32986176 0.39145139 0.451556 0.51014116 0.56717407;0.66382255 0.76674007 0.87195994 0.97902709 1.08751345;0.74192008 0.90592151 1.07717006 1.25120233 1.42395676)

expectedH:(0.24635157 0.28610883 0.32240467 0.35525807 0.38474904;0.49223563 0.55611431 0.61507696 0.66844003 0.7159181;0.56735664 0.66310127 0.74419266 0.80889665 0.858299)

lg "check relative error of simple step forward:"
relError[expectedH;res 0]
relError[expectedC;res 1]

lg "##############################
    LSTM step backward
    ##############################"

`N`D`H set' 4 5 6
x: rad N,D
prevH: rad N,H
prevC: rad N,H
wx: rad D,4*H
wh: rad H,4*H
b: rad 4*H

d:`x`prevH`prevC`wx`wh`b!(x;prevH;prevC;wx;wh;b)
nextHnextCCache:lstmStepForward d
dnextH: rad shape nextHnextCCache 0
dnextC: rad shape nextHnextCCache 1

lg "now get numerical grads for comparison"
numGrad:{[x]numericalGradientArray[(first lstmStepForward@);d;dnextH;x]+numericalGradientArray[(@[;1] lstmStepForward@);d;dnextC;x]}
numGrads:numGrad each `x`prevH`prevC`wx`wh`b

grads: lstmStepBackward[dnextH;dnextC;nextHnextCCache 2]

lg "relative errors are:"
lg key[grads]!relError'[value grads;numGrads]

lg "##############################
    LSTM forward
    ##############################"

lg "we now run the lstm forward on an entire timeseries and do a grad check"

`N`D`H`T set' 2 5 4 3
x:(N;T;D)#linSpace[-0.4;0.6;N*T*D]
h0:(N;H)#linSpace[-0.4;0.8;N*H]
wx:(D;4*H)#linSpace[-0.2;0.9;D*4*H]
wh:(H;4*H)#linSpace[-0.3;0.6;H*4*H]
b:linSpace[0.2;0.7;4*H]

expectedH:((0.01764008 0.01823233 0.01882671 0.0194232;0.1128749 0.1214623 0.1301845 0.1390294;0.3135877 0.3333863 0.3530445 0.3725097);(0.4576788 0.4761092 0.4936887 0.5104194;0.6704845 0.6935009 0.7148601 0.7346449;0.8173351 0.8367787 0.8540375 0.8693531))

hCache:lstmForward `x`h0`wx`wh`b!(x;h0;wx;wh;b)

lg "relative h error is "
relError[expectedH;hCache 0]

lg "##############################
    LSTM backward
    ##############################"

lg "we now implement the backward pass for LSTM over oan entire timeseries of data,
    then we do a gradient check, errors should be approx 1e-8 or less"

`N`D`T`H set'2 3 10 6;

x:rad N, T, D
h0:rad N, H
wx:rad D, 4*H
wh:rad H*1 4
b:rad 4*H
outCache:lstmForward `x`h0`wx`wh`b!(x;h0;wx;wh;b)
dout:rad shape outCache 0;
grads:lstmBackward[dout;outCache 1]

lg "now get numerical grads for comparison "
d:`x`h0`wx`wh`b!(x;h0;wx;wh;b)
dxNum:numericalGradientArray[(first lstmForward@);d;dout;`x]
dh0Num:numericalGradientArray[(first lstmForward@);d;dout;`h0]
dwxNum:numericalGradientArray[(first lstmForward@);d;dout;`wx]
dwhNum:numericalGradientArray[(first lstmForward@);d;dout;`wh]
dbNum:numericalGradientArray[(first lstmForward@);d;dout;`b]

lg key[grads]!relError'[value grads;(dxNum;dh0Num;dwxNum;dwhNum;dbNum)]

lg "##############################
    LSTM captioning model
    ##############################"

`N`D`W`H set' 10 20 30 40
wordToIdx:(nullToken,`cat`dog)!0 2 3
V:count wordToIdx
T:13
startd:(!). flip((`wordToIdx;wordToIdx);`dimInput,D;`dimWordVec,W;`dimHidden,H;`cellType`lstm);
initd:captioningRNN.init startd;

/ scale everything to a linSpace of that shape
initd:@[initd;captioningRNN.params[];{shapex#linSpace[-1.4;1.3;prd shapex:shape x]}]

features:(N;D)#linSpace[-0.5;1.7;N*D]
captions:(N,T)#til[N*T]mod V;

lossGrads:captioningRNN.loss @[initd;`features`captions;:;(features;captions)]

lg "comparing loss to expected "
lg "loss is ",string loss:first lossGrads
lg "expected loss is ",string expectedLoss:9.82445935443
lg "relative error is "
relError[loss;expectedLoss]

lg "##############################
    Overfit small data
    ##############################"

lg "we now overfit a model using a small sample of 100 training examples,
    we should see losses around 1 after 60 epochs"

/ temporary
mask:-50?count train_captions
smallData:(!). flip ((`trainCaptions;train_captions mask);(`trainImageIdxs;train_image_idxs mask);(`valCaptions;val_captions);(`valImageIdxs;val_image_idxs);(`trainFeatures;train_features);(`valFeatures;val_features);(`idxToWord;idx_to_word);(`wordToIdx;word_to_idx);(`trainUrls;train_urls);(`valUrls;val_urls));


startd:smallData,(!). flip (`updateRule`adam;`numEpochs,50;`batchSize,25;(`optimConfig;enlist[`learnRate]!enlist 5e-3);`lrDecay,0.05;`printEvery,10;`cellType`lstm;(`wordToIdx;word_to_idx);(`dimInput;count train_features 0);`dimHidden,512;`dimWordVec,256;`model`captioningRNN)

if[runAll;res:captioningSolver.train startd

    lg "plot loss history, validation and training accuracy in an IDE e.g qstudio using scatterplots:"
    lg"loss history: ([]iteration:til count res`lossHistory;loss:res`lossHistory)"
  ];
lg "##############################
    Test time sampling
    ##############################"

if[runAll;sampleCaptions[smallData;res;] each `train`val];

lg "##############################
    Train a good model
    #############################"

lg "we now try and train a decent model, hopefully we will see better results
    than the random stuff from the overfit model earlier"

mask:-10000?count train_captions
smallData:(!). flip ((`trainCaptions;train_captions mask);(`trainImageIdxs;train_image_idxs mask);(`valCaptions;val_captions);(`valImageIdxs;val_image_idxs);(`trainFeatures;train_features);(`valFeatures;val_features);(`idxToWord;idx_to_word);(`wordToIdx;word_to_idx);(`trainUrls;train_urls);(`valUrls;val_urls));


startd:smallData,(!). flip (`updateRule`adam;`numEpochs,10;`batchSize,25;(`optimConfig;enlist[`learnRate]!enlist 5e-3);`lrDecay,0.995;`printEvery,10;`cellType`lstm;(`wordToIdx;word_to_idx);(`dimInput;count train_features 0);`dimHidden,512;`dimWordVec,256;`model`captioningRNN)

res:captioningSolver.train startd

sampleCaptions[smallData;res;] each `train`val
