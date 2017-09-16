\l nn_util.q
\c 20 150
ixToChar:chars:distinct data:"\n"sv read0`input.txt
dataSize:count data;
-1"total characters+uniques are ",-3!dataSize,vocabSize:count chars;
charToIx:chars!til count chars;
/ hyperparams
hiddenSize:100;
seqLength:25;
learningRate:0.1;
/ model params
wxh:0.01*rad hiddenSize,vocabSize; // input to hidden
whh:0.01*rad 2#hiddenSize; // hidden to hidden
why:0.01*rad vocabSize,hiddenSize; // hidden to output
bh:hiddenSize#0f; // hidden bias
bY:vocabSize#0f;                                                                     // output bias
lossFunc:{[inputs;targets;hprev]
    res:(c:count inputs){[d] i:d`i;
         d[`xs],:enlist xs:@[d[`vocabSize]#0;d[`inputs;i];:;1]; // forward pass
         d[`hs],:enlist hs:tanh[dot[wxh;"f"$xs]+dot["f"$whh;"f"$last d`hs]+bh];
         d[`ys],:enlist ys:dot[why;hs]+bY;
         d[`ps],:enlist ps:e%sumo e:exp ys;
         @[@[d;`loss;-;log first ps d[`targets;i]];`i;1+]
     }/`i`vocabSize`inputs`targets`xs`hs`ys`ps`loss!(0;vocabSize;inputs;targets;();enlist hprev;();();0f);
     res:c{[d] ind:seqLength-i:d`i;                                                               // backward pass
         d[`dby]+:dy:@[d[`ps;i];d[`targets;i];-;1];
         d[`dwhy]+:dot[enlist each dy;enlist dhs:d[`hs;i+1]];
         dh:dot[flip why;dy]+d`dhnext;
         d[`dbh]+:dhraw:dh*1-{x*x}dhs;
         d[`dwxh]+:dot[edhraw:enlist each dhraw;enlist "f"$d[`xs;i]];
         d[`dwhh]+:dot[edhraw;enlist d[`hs;i]];
         d[`dhnext]:dot[flip whh;dhraw];
         @[d;`i;-;1]
     }/@[res;`i;:;c-1],`dwxh`dwhh`dwhy`dbh`dby`dhnext!(wxh;whh;why;bh;bY;res[`hs;0])*0f;
     res:@[res;`dwxh`dwhh`dwhy`dbh`dby;-5|5&]; // bound params between (-5;5), reverse
    `loss`dwxh`dwhh`dwhy`dbh`dby`hs#@[res;`hs;last]
 };
sample:{[h;seedIx;n]
     n{[d] d[`h]:tanh[dot[wxh;"f"$d`x]+dot[whh;d`h]]+bh;
           p:{x%sumo x}exp dot[why;d`h]+bY;
           ix:randChoiceP[til vocabSize;1;p];
           d:@[d;`x;:;@[vocabSize#0;ix;:;1]];
           @[d;`ixes;,;ix]
      }/`h`x`ixes!(h;@[vocabSize#0;seedIx;:;1];())
 };
n:p:0;
@[`.;`mwxh`mwhh`mwhy`mbh`mby;:;mem:(wxh;whh;why;bh;bY)*0f];
smoothLoss:neg seqLength*log 1%vocabSize
\l prof.q
dot:{[x;y].qml.mm[x;y]}
.prof.instrall[]
while[1b;
    if[any(n=0;(p+seqLength+1)>=count data);
        -1"resetting hprev and p";
        hprev:hiddenSize#0f;
        p:0];
    inputs:charToIx data ind:p+til seqLength;
    targets:charToIx data 1+ind;
    if[0=n mod 100;
        sampleIx:sample[hprev;inputs 0;200]`ixes;
        -1"----\n ",chars[sampleIx],"\n----"];
    lossGrads:lossFunc[inputs;targets;hprev];
    hprev:lossGrads`hs;
    smoothLoss:(0.001*lossGrads`loss)+smoothLoss*0.999;
    if[0=n mod 100;-1"iteration ",string[n]," loss: ",string smoothLoss];
    dparam:lossGrads`dwxh`dwhh`dwhy`dbh`dby;
    mem:mem+dparam*dparam;
    `wxh`whh`why`bh`bY set'(wxh;whh;why;bh;bY)-learningRate*dparam%sqrt mem+1e-8;
    n+:1;
    p+:seqLength;
 ];
