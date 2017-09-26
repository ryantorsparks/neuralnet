@[system;"p 5000";{-1"WARNING: failed to set port to 5000";}]
\l load_all.q
\l load_coco_data.q
/ set runAll to 1b only if you want to run everything
runAll:0b

lg "##############################
    Recurrent Neural Networks
    ##############################"

lg "##############################
    Vanilla RNN - step forward
    ##############################"

@[`.;`N`D`H;:;3 10 4];
x:(N,D)#linSpace[-0.4;0.7;N*D]
prevH:(N;H)#linSpace[-0.2;0.5;N*H]
wx:(D,H)#linSpace[-0.1;0.9;D*H]
wh:(H,H)#linSpace[-0.3;0.70;H*H]
b:linSpace[-0.2;0.4;H]
res:rnnStepForward[x;prevH;wx;wh;b]
expectedH:(-0.5817209 -0.5018203 -0.4123277 -0.314101;0.6685469 0.7956238 0.8775555 0.9279597;0.979345 0.9914421 0.9964669 0.9985435)
lg "check relative error of simple step forward:"
relError[expectedH;res 0]


lg "##############################
    Vanilla RNN - step backward
    ##############################"

@[`.;`N`D`H;:;4 5 6];
x:rad N,D
h:rad N,H
wx:rad D,H
wh:rad H,H
b:rad H

outCache:rnnStepForward[x;h;wx;wh;b]
dnexth:rad shape outCache 0

lg "now get numerical grads for comparison"
dxNum:numericalGradientArray[(first rnnStepForward[;h;wx;wh;b]@);x;dnexth;`x]
dprevHNum:numericalGradientArray[(first rnnStepForward[x;;wx;wh;b]@);h;dnexth;`h]
dwxNum:numericalGradientArray[(first rnnStepForward[x;h;;wh;b]@);wx;dnexth;`wx]
dwhNum:numericalGradientArray[(first rnnStepForward[x;h;wx;;b]@);wh;dnexth;`wh]
dbNum:numericalGradientArray[(first rnnStepForward[x;h;wx;wh;]@);b;dnexth;`b]

grads:rnnStepBackward[dnexth;outCache 1]

relError'[value grads;(dxNum;dprevHNum;dwxNum;dwhNum;dbNum)]

lg "##############################
    Vanilla RNN - forward
    ##############################"

lg "we now implement a RNN that process an entire sequence of data."

@[`.;`N`T`D`H;:;2 3 4 5];
x:(N;T;D)#linSpace[-0.1;0.3;N*T*D]
h0:(N;H)#linSpace[-0.3;0.1;N*H]
wx:(D;H)#linSpace[-0.2;0.4;D*H]
wh:(H;H)#linSpace[-0.4;0.1;H*H]
b:linSpace[-0.7;0.1;H]

hCache:rnnForward `x`h0`wx`wh`b!(x;h0;wx;wh;b)

expectedH:((-0.4207075 -0.2727926 -0.1107494 0.05740409 0.2223625;-0.3952581 -0.2255466 -0.0409454 0.1464941 0.3239732;-0.4230511 -0.2422373 -0.04287027 0.1599704 0.3501453);(-0.5585747 -0.3906582 -0.1919818 0.02378408 0.2373567;-0.271502 -0.07088804 0.1356294 0.3309973 0.5015877;-0.5101482 -0.3052443 -0.06755202 0.1780639 0.4033304))

lg "relative error compared to expected h "
relError[hCache 0;expectedH]

lg "##############################
    Vanilla RNN - backward
    ##############################"

@[`.;`N`D`T`H;:;2 3 10 5];

x:rad N, T, D
h0:rad N, H
wx:rad D, H
wh:rad H, H
b:rad H
outCache:rnnForward `x`h0`wx`wh`b!(x;h0;wx;wh;b)
dout:rad shape outCache 0;
grads:rnnBackward[dout;outCache 1]

lg "now get numerical grads for comparison "

dxNum:numericalGradientArray[(first rnnForward@);`x`h0`wx`wh`b!(x;h0;wx;wh;b);dout;`x]
dh0Num:numericalGradientArray[(first rnnForward@);`x`h0`wx`wh`b!(x;h0;wx;wh;b);dout;`h0]
dwxNum:numericalGradientArray[(first rnnForward@);`x`h0`wx`wh`b!(x;h0;wx;wh;b);dout;`wx]
dwhNum:numericalGradientArray[(first rnnForward@);`x`h0`wx`wh`b!(x;h0;wx;wh;b);dout;`wh]
dbNum:numericalGradientArray[(first rnnForward@);`x`h0`wx`wh`b!(x;h0;wx;wh;b);dout;`b]

relError'[value grads;(dxNum;dh0Num;dwxNum;dwhNum;dbNum)]


lg "##############################
    Word embedding - forward
    ##############################"

lg "In deep learning systems, we commonly represent words using vectors.
    Each word of the vocabulary will be associated with a vector, and 
    these vectors will be learned jointly with the rest of the system."

@[`.;`N`T`V`D;:;2 4 5 3];
x:2 4#0 3 1 2 2 1 0 3
w:(V;D)#linSpace[0;1;V*D]

out:first wordEmbeddingForward[x;w]

expectedOut:((0 0.07142857 0.1428571;0.6428571 0.7142857 0.7857143;0.2142857 0.2857143 0.3571429;0.4285714 0.5 0.5714286);(0.4285714 0.5 0.5714286;0.2142857 0.2857143 0.3571429;0 0.07142857 0.1428571;0.6428571 0.7142857 0.7857143))

lg "compare forward pass against expected value: "
relError[out;expectedOut]

lg "##############################
    Word embedding - backward
    ##############################"

@[`.;`N`T`V`D;:;50 3 5 6];

x:(N;T)#(N*T)?V;

w:rad V,D

outCache:wordEmbeddingForward[x;w];

dout:rad shape outCache 0;

dw:wordEmbeddingBackward[dout;outCache 1];

dwNum:numericalGradientArray[(first wordEmbeddingForward[x;]@);w;dout;`w];

lg "relative error compared to numerical gradient"
relError[dw;dwNum]

lg "##############################
    Temporal affine layer
    ##############################"

lg "At every timestep we use an affine function to transform the RNN hidden 
    vector at that timestep into scores for each word in the vocabulary"

@[`.;`N`T`D`M;:;2 3 4 5];

x:rad N,T,D;
w:rad D,M;
b:rad M;

outCache:temporalAffineForward[x;w;b];
dout:rad shape outCache 0;

dxNum:numericalGradientArray[(first temporalAffineForward[;w;b]@);x;dout;`x];
dwNum:numericalGradientArray[(first temporalAffineForward[x;;b]@);w;dout;`w];
dbNum:numericalGradientArray[(first temporalAffineForward[x;w;]@);b;dout;`b];

grads: temporalAffineBackward[dout;outCache 1];

lg "relative error compared to numerical gradient"
relError'[value grads;(dxNum;dwNum;dbNum)]


lg "##############################
    Temporal softmax loss
    ##############################"

lg "in an RNN language model, at every timestep we produce a score for 
    each word in the vocabulary. We know the ground-truth word at each timestep, 
    so we use a softmax loss function to compute loss and gradient at each 
    timestep. We sum the losses over time and average them over the minibatch. 
    However, since we operate over minibatches and different captions may have 
    different lengths, we append <NULL> tokens to the end of each caption so 
    they all have the same length. We don't want these <NULL> tokens to count 
    toward the loss or gradient, so in addition to scores and ground-truth 
    labels our loss function also accepts a mask array that tells it which 
    elements of the scores count towards the loss."

@[`.;`N`T`V;:;100 1 10];

checkLoss:{[N;T;V;p]
    x:0.001*rad N,T,V;
    y:(N;T)#(N*T)?V;
    mask:p>=(N;T)#(N*T)?1.0;
    first temporalSoftmaxLoss[x;y;mask]
 };

lg "run a few random loss checks, losses should be approx 2.3, 23, 2.3"
checkLoss'[100 100 5000;1 10 10;10 10 10; 1.0 1.0 0.1]

lg "##############################
    RNN for imaging captioning
    ##############################"

@[`.;`N`D`W`H`T;:;10 20 30 40 13];
wordToIdx:(nullToken,`cat`dog)!0 2 3;
V:count wordToIdx;

startd:(!). flip((`wordToIdx;wordToIdx);`dimInput,D;`dimWordVec,W;`dimHidden,H;`cellType`rnn);
initd:captioningRNN.init startd;

/ scale everything to a linSpace of that shape
initd:@[initd;captioningRNN.params[];{shapex#linSpace[-1.4;1.3;prd shapex:shape x]}]

features:(N;D)#linSpace[-1.5;0.3;N*D]
captions:(N,T)#til[N*T]mod V;

lossGrads:captioningRNN.loss @[initd;`features`captions;:;(features;captions)]
/
lg "comparing loss to expected "
lg "loss is ",string loss:first lossGrads
lg "expected loss is ",string expectedLoss:9.83235591003
lg "relative error is "
relError[loss;expectedLoss]

lg "we now check numerical gradients"
wordToIdx:(nullToken,`cat`dog)!0 2 3;
startd:(!). flip (`batchSize,2;`timesteps,3;`dimInput,4;`dimWordVec,5;`dimHidden,6;(`wordToIdx;wordToIdx);`vocabSize,count wordToIdx)
startd[`captions`features]:({(x;y)#(x*y)?z}[startd`batchSize;startd`timesteps;startd`vocabSize];rad startd`batchSize`dimInput)

initd:captioningRNN.init startd
lossGrads:captioningRNN.loss initd

lg "relative errors are "

captioningRNN.params[]!{[grads;initd;param] relError[grads param;numericalGradientArray[(first captioningRNN.loss@);initd;initd param;param]]}[lossGrads 1;@[initd;`h;:;1e-6]]each captioningRNN.params[]
\

lg "##############################
    Overfit small data
    ##############################"

lg "we now overfit a model using a small sample of 100 training examples,
    we should see losses around 1 after 50 epochs"

/ temporary
mask:get `:rnnOverfitMask
smallData:(!). flip ((`trainCaptions;train_captions mask);(`trainImageIdxs;train_image_idxs mask);(`valCaptions;val_captions);(`valImageIdxs;val_image_idxs);(`trainFeatures;train_features);(`valFeatures;val_features);(`idxToWord;idx_to_word);(`wordToIdx;word_to_idx);(`trainUrls;train_urls);(`valUrls;val_urls));


startd:smallData,(!). flip (`updateRule`adam;`numEpochs,50;`batchSize,25;(`optimConfig;enlist[`learnRate]!enlist 5e-3);`lrDecay,0.05;`printEvery,10;`cellType`rnn;(`wordToIdx;word_to_idx);(`dimInput;count train_features 0);`dimHidden,512;`dimWordVec,256;`model`captioningRNN)

res:captioningSolver.train startd

lg "plot loss history, validation and training accuracy in an IDE e.g qstudio using scatterplots:"
lg"loss history: ([]iteration:til count res`lossHistory;loss:res`lossHistory)"

lg "##############################
    Test time sampling
    ##############################"

lg "we now compare ground truth captions against the result from our overfitting training,
    and also against validation data that we caption from our overfit training. Threfore we 
    expect our training captions to be very accurate, but our validation captions to be 
    garbage."

sampleCaptions[smallData;res;] each `train`val;
