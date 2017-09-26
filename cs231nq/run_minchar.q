//system"S ",string `int$.z.t
system"l qml.q";dot:.qml.mm;tanh:.qml.tanh                                                                                  // mmu, tanh funcs
rad:{[dims](dims)#sqrt[-2*log n?1.]*cos[2*3.1415926535897931*(n:prd dims)?1.]}                                              // random, normally distributed n dim matrix
randChoiceP:{[n;c;p] n@sums[p] binr c?1f}                                                                                   // same as np.random.choice
charToIx:chars!til count chars:distinct data:"\n"sv read0`input.txt                                                         // read in text file, init dicts
dataSize:count data;
-1"total characters+uniques are ",-3!dataSize,vocabSize:count chars;
//`hiddenSize`seqLength`learningRate set'(20+rand 100),(10+rand 50),0.05+rand 0.1;                                                                          // hyperparams
`hiddenSize`seqLength`learningRate set'100,25,0.1;                                                                          // hyperparams
`wxh`whh`why`bh`bY set'0.01*(rad hiddenSize,vocabSize;rad 2#hiddenSize;rad vocabSize,hiddenSize;hiddenSize#0f;vocabSize#0f) // model params
`fwhy`fwhh set' flip each (why;whh)                                                                                         // cache the flips
fwd:{[d] i:d`i;                                                                                                             //   of hidden states (float). Returns loss+grads
         d[`xs],:enlist xs:@[d[`vocabSize]#0f;d[`inputs;i];:;1f];                                                           // forward pass
         d[`hs],:enlist hs:tanh dot[wxh;xs]+dot[whh;last d`hs]+bh;
         d[`ys],:enlist ys:dot[why;hs]+bY;                                                                                  // unnormalized log probabilities for next chars
         d[`ps],:enlist ps:e%(sum/) e:exp ys;                                                                               // probabilities for next chars
         @[@[d;`loss;-;log first ps d[`targets;i]];`i;1+]};                                                                 // softmax loss
back:{[d] i:d`i; 
         d[`dby]+:dy:@[d[`ps;i];d[`targets;i];-;1];
         d[`dwhy]+:dy*\:dhs:d[`hs;i+1];
         dh:dot[fwhy;dy]+d`dhnext;
         d[`dbh]+:dhraw:dh*1-{x*x}dhs;                                                                                      // backprop through tanh nonlinearity
         d[`dwxh]+:dhraw*\:d[`xs;i];
         d[`dwhh]+:dhraw*\:d[`hs;i];
         d[`dhnext]:dot[fwhh;dhraw];
         @[d;`i;-;1]}
lossFunc:{[inputs;targets;hprev]                                                                                            // inputs/targets are longs, hprev is list
    res:(c:count inputs)fwd/`i`vocabSize`inputs`targets`xs`hs`ys`ps`loss!(0;vocabSize;inputs;targets;();enlist hprev;();();0f);
    res:c back/@[res;`i;:;c-1],`dwxh`dwhh`dwhy`dbh`dby`dhnext!(wxh;whh;why;bh;bY;res[`hs;0])*0f;                            
    res:@[res;`dwxh`dwhh`dwhy`dbh`dby;-5|5&];                                                                              // bound params between (-5;5)
    `loss`dwxh`dwhh`dwhy`dbh`dby`hs#@[res;`hs;last]};
sample:{[h;seedIx;n]                                                                                                        // sample a seq of longs from model, h is memory
     n{[d] d[`h]:tanh dot[wxh;d`x]+dot[whh;d`h]+bh;                                                                         //   state, seedIx is seed letter for first timestep
         d[`ixes],:ix:randChoiceP[til vocabSize;1;{x%(sum/) x}exp dot[why;d`h]+bY];
         @[d;`x;:;@[vocabSize#0;ix;:;1]]}/`h`x`ixes!(h;@[vocabSize#0;seedIx;:;1];())};
`now`smoothLoss`n`p`mwxh`mwhh`mwhy`mbh`mby set'(.z.t;neg seqLength*log 1%vocabSize;0;0),mem:(wxh;whh;why;bh;bY)*0f;         // init memories (for adagrad update) + counter
while[1b;                                                                                                                   // keep running forever
    if[any(n=0;(p+seqLength+1)>=count data);-1"resetting hprev and p";hprev:hiddenSize#0f;p:0];
    inputs:charToIx data ind:p+til seqLength;
    targets:charToIx data 1+ind;
    if[0=n mod 100;-1"----\n ",chars[sample[hprev;inputs 0;200]`ixes],"\n----"];                                            // sample text every 100 iters
    lossGrads:lossFunc[inputs;targets;hprev];                                                                               // forward seqLength chars through and get grad
    hprev:lossGrads`hs;
    smoothLoss:(0.001*lossGrads`loss)+smoothLoss*0.999;
    if[0=n mod 100;-1 string[.z.t-now]," iteration ",string[n]," loss: ",string smoothLoss;now:.z.t];                                                   // track loss
    dparam:lossGrads`dwxh`dwhh`dwhy`dbh`dby;
    `wxh`whh`why`bh`bY set'(wxh;whh;why;bh;bY)-learningRate*dparam%sqrt 1e-8+mem+:{x*x}dparam;                              // adagrad update
    `fwhy`fwhh set'flip each (why;whh);
    p+:seqLength;n+:1];
