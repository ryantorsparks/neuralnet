\l nn_util.q
charToIx:chars!til count chars:distinct data:"\n"sv read0`input.txt                                                         // read in text file, init dicts
dataSize:count data;
-1"total characters+uniques are ",-3!dataSize,vocabSize:count chars;
`hiddenSize`seqLength`learningRate set'100,25,0.1;                                                                          // hyperparams
`wxh`whh`why`bh`bY set'0.01*(rad hiddenSize,vocabSize;rad 2#hiddenSize;rad vocabSize,hiddenSize;hiddenSize#0f;vocabSize#0f) // model params
lossFunc:{[inputs;targets;hprev]                                                                                            // inputs/targets are longs, hprev is list
    res:(c:count inputs){[d] i:d`i;                                                                                         //   of hidden states (float). Returns loss+grads
         d[`xs],:enlist xs:@[d[`vocabSize]#0;d[`inputs;i];:;1];                                                             // forward pass
         d[`hs],:enlist hs:tanh dot[wxh;"f"$xs]+dot["f"$whh;"f"$last d`hs]+bh;
         d[`ys],:enlist ys:dot[why;hs]+bY;                                                                                  // unnormalized log probabilities for next chars
         d[`ps],:enlist ps:e%sumo e:exp ys;                                                                                 // probabilities for next chars
         @[@[d;`loss;-;log first ps d[`targets;i]];`i;1+]                                                                   // softmax loss
       }/`i`vocabSize`inputs`targets`xs`hs`ys`ps`loss!(0;vocabSize;inputs;targets;();enlist hprev;();();0f);
     res:c{[d] i:d`i;                                                                                                       // backward pass
         d[`dby]+:dy:@[d[`ps;i];d[`targets;i];-;1];
         d[`dwhy]+:dot[enlist each dy;enlist dhs:d[`hs;i+1]];
         dh:dot[flip why;dy]+d`dhnext;
         d[`dbh]+:dhraw:dh*1-{x*x}dhs;                                                                                      // backprop through tanh nonlinearity
         d[`dwxh]+:dot[edhraw:enlist each dhraw;enlist "f"$d[`xs;i]];
         d[`dwhh]+:dot[edhraw;enlist d[`hs;i]];
         d[`dhnext]:dot[flip whh;dhraw];
         @[d;`i;-;1]
       }/@[res;`i;:;c-1],`dwxh`dwhh`dwhy`dbh`dby`dhnext!(wxh;whh;why;bh;bY;res[`hs;0])*0f;
     res:@[res;`dwxh`dwhh`dwhy`dbh`dby;-5|5&];                                                                              // bound params between (-5;5), reverse
    `loss`dwxh`dwhh`dwhy`dbh`dby`hs#@[res;`hs;last]};
sample:{[h;seedIx;n]                                                                                                        // sample a seq of longs from model, h is memory
     n{[d] d[`h]:tanh dot[wxh;"f"$d`x]+dot[whh;d`h]+bh;                                                                     //   state, seedIx is seed letter for first timestep
           d[`ixes],:ix:randChoiceP[til vocabSize;1;{x%sumo x}exp dot[why;d`h]+bY];
           @[d;`x;:;@[vocabSize#0;ix;:;1]]
      }/`h`x`ixes!(h;@[vocabSize#0;seedIx;:;1];())};
`smoothLoss`n`p`mwxh`mwhh`mwhy`mbh`mby set'(neg seqLength*log 1%vocabSize;0;0),mem:(wxh;whh;why;bh;bY)*0f;                  // init memories (for adagrad update) + counter
while[1b;                                                                                                                   // keep running forever
    if[any(n=0;(p+seqLength+1)>=count data);-1"resetting hprev and p";hprev:hiddenSize#0f;p:0];
    inputs:charToIx data ind:p+til seqLength;
    targets:charToIx data 1+ind;
    if[0=n mod 100;-1"----\n ",chars[sample[hprev;inputs 0;200]`ixes],"\n----"];                                            // sample text every 100 iters
    lossGrads:lossFunc[inputs;targets;hprev];                                                                               // forward seqLength chars through and get grad
    hprev:lossGrads`hs;
    smoothLoss:(0.001*lossGrads`loss)+smoothLoss*0.999;
    if[0=n mod 100;-1"iteration ",string[n]," loss: ",string smoothLoss];                                                   // track loss
    dparam:lossGrads`dwxh`dwhh`dwhy`dbh`dby;
    `wxh`whh`why`bh`bY set'(wxh;whh;why;bh;bY)-learningRate*dparam%sqrt 1e-8+mem+:{x*x}dparam;                              // adagrad update
    p+:seqLength;n+:1];
