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


