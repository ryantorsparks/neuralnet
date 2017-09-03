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
