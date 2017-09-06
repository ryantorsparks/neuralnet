/ ###################### recurrent neurlal network layers  #######################

/ forward pass function for a single timestep of a vanilla
/ RNN, using a tanh activation function. Input data has dimension D, hidden state
/ has dimension H, minibiatch of size N is used
/ inputs are:
/   x: input data, shape (N;D)
/   prevH: state from previous timestep, shape (N;D)
/   wx: weight matrix for input to hidden connecitons, shape (D;H)
/   wh: weight matrix for hidden to hidden connections, shape (H;H)
/   b: bias, list length H
/ returns: (nextH; cache), where:
/   nextH: next hidden state, shape (N;H)
/   cache: dict of values needed for back pass
rnnStepForward:{[x;prevH;wx;wh;b]
    forward:dot[x;wx]+dot[prevH;wh]+\:b;

    / squash using tanh
    nextH:tanh forward;

    / return result with cache
    cache: `x`wx`prevH`wh`forward!(x;wx;prevH;wh;forward);
    (nextH;cache)
 };

/ backward pass func for a single timestep of vanilla RNN
/ inputs are:
/   dnextH: gradient of loss wrt next hidden state
/   cache: cache from forward pass (dict, key should be `x`wx`prevH`wh`forward)
/ Returns a dict, with elements:
/   `dx: gradients of input data, shape (N;D)
/   `dprevH: gradients of previous hidden stae, shape (N;H)
/   `dwx: gradients of input-to-hidden weights, shape (N;H)
/   `dwh: gradients of hidden-to-hidden weights, shape (H;H)
/   `db: gradients of bias vector, list of length H
rnnStepBackward:{[dnextH;cache]

    / backprop step 2
    dforward:dnextH*1-{x*x}tanh cache`forward;
    
    / backprop step 1
    dx:dot[dforward;flip cache`wx];
    dwx:dot[flip cache`x;dforward];
    dprevH:dot[dforward;flip cache`wh];
    dwh:dot[flip cache`prevH;dforward];
    db:sum dforward;
   
    `dx`dprevH`dwx`dwh`db!(dx;dprevH;dwx;dwh;db)
 };
  

/ function to run a vanilla RNN forward on an entire sequence of data
/ input sequence is dimension (T;D), RNN uses a hidden size of H, and we
/ work over a minibatch which contains N sequences. After running the RNN
/ forward, we return the hidden states for all timesteps
/ inputs are:
/   x: input data for entire timeseries, shape (N;T;D)
/   h0: initial staet, shape (N;H)
/   wx: weight matrix for input-to-hidden connections, shape (D;H)
/   wh: weight matrix for hidden-to-hidden connections, shape (H;H)
/   b: bias, list length H
/ returns (h;cache):
/   h: hiden states for entire timeseries, shape (N;T;H)
/   cache: list of dicts of vals needed for back pass, i.e. a table
/          with cols ([]x;wx;prevH;wh;forward)
rnnForward:{[d]
    x:d`x;
    shapex:shape x;
    N:shapex 0;
    T:shapex 1;
    D:shapex 2;
    h0:d`h0;
    H:count first h0;
    cache:()!();
   
    x:flip x;
    h:(T;N;H)#0f;
    h:@[h;2;:;h0];

    rnnForwardLoop:{[hCacheList;x;wx;wh;b]
    //    show(hCacheList;x);
        hlist:last hCacheList 0;
        cachelist:hCacheList 1;
        res:rnnStepForward[x;hlist;wx;wh;b];
        hCacheList,'enlist each res
     };
    hCacheList:rnnForwardLoop[;;d`wx;d`wh;d`b]/[(enlist h0;flip`x`wx`prevH`wh`forward!());x];
    (flip 1_ hCacheList 0;hCacheList 1)
 };

/ backward pass function for vanilla RNN over entire sequence
/ of data
/ inputs are
/   dh: upstream grads, shape (N;T;H)
/   cache: table (list of dicts) from rnnForward, cols are `x`wx`prevH`wh`forward
/ returns a cache dict, with elements:
/   `dx: gradients of inputs, shape (N;T;D)
/   `dh0: gradient of initial hidden state, shape (N;H)
/   `dwx: gradient of input-to-hidden weights, shape (D;H)
/   `dwh: gradient of hidden-to-hidden weights, shape (H;H)
/   `db: gradient of biases, list length H
rnnBackward:{[dh;cacheList]
    shapeDh:shape dh;
    N:shapeDh 0;
    T:shapeDh 1;
    H: shapeDh 2;

    / shape[1] of the first x in cache
    D:count cacheList[0;`x;0];
   
    dhList:flip dh;
    grads:`dh0`db`dwh`dwx!(N,H;H;H,H;D,H)#\:0f;
    grads[`dxList`dprevH]:(();(N;H)#0f);

    rnnBackwardLoop:{[d;dh;cache]
        dh+:d`dprevH;
        grads:rnnStepBackward[dh;cache];
        d[`dxList]:enlist[grads`dx],d`dxList;
        d[`dprevH]:grads`dprevH;
        d+:`dx`dprevH _ grads;
        d
     };

    res:rnnBackwardLoop/[grads;reverse dhList;reverse cacheList]; 
    `dx`dh0`dwx`dwh`db!(flip res`dxList;res`dprevH;res`dwx;res`dwh;res`db)
 };

/ word embedding function forward pass
/ operate on mini batches of size N, with each sequence
/ length T. We assume a vocabulary of V words, assigning each a vector of
/ dimension dimension D
/ inputs:
/   x: integer matrix, shape (N;T), giving indices of words. Each element
/      of x must be in range 0<=idx<V
/   w: weight matrix shape (V;D), giving word vectors for all words
/ returns (out;cache), where:
/   out: array shape (N;T;D) giving word vectors for all input words
/   cache: dict of vals for back pass `x`w
wordEmbeddingForward:{[x;w]
    out:w x;
    cache: `x`w!(x;w);
    (out;cache)
 };

/ word embedding function backward pass
/ we cannot back propagate into the words as they are ints,
/ so we only return gradient for the word embedding matrix
/ inputs:
/   dout: upstream grads, shape (N;T;D)
/   cache: vals from forward pass, dict with key `x`w
/ returns:
/   dw: gradient of word embedding matrix, shape (V;D)
wordEmbeddingBackward:{[dout;cache]
    x:cache`x;
    w:cache`w;
    @[;;+;]/[w*0;x;dout]
 };

/ temporal affine forward function
/ forward pass func, input is a set of D-dimensional vectors arranged into 
/ a minibatch of N timeseries, each of length T. Use affine func to tranform
/ each of those vetors to a new vector of dimension M
/ inputs:
/   x:input data shape (N;T;D)
/   w: weights, shape (D;M)
/   b: biases, list length M
/ returns: (out;cache):
/   out: shape (N;T;M)
/   cache: dict, `x`w`b`out for key
temporalAffineForward:{[x;w;b]
    shapeX:shape x;
    N:shapeX 0;
    T:shapeX 1;
    D:shapeX 2;
    M:count b;
    out:b+/:/:reshapeM[dot[reshapeM[x;(N*T;D)];w];(N;T;M)];
    cache:`x`w`b`out!(x;w;b;out);
    (out;cache)
 };

/ backward pass func for temporal affine filter
/ input:
/   dout: upstream grads, shape (N;T;M)
/   cache: dict from forward pass, key is `x`w`b`out
/ returns, dict:
/   `dx: grads, shape (N;T;D)
/   `dw: grads of weight, shape (D;M)
/   `db: grads of biases, list length M
temporalAffineBackward:{[dout;cache]
    x:cache`x;
    w:cache`w;
    b:cache`b;
    out:cache`out;
    shapeX:shape x;
    N:shapeX 0;
    T:shapeX 1;
    D:shapeX 2;
    M:count b;

    dx:reshapeM[ dot[reshapeM[dout;(N*T;M)];flip w]; (N;T;D)];
    dw:flip dot[flip[reshapeM[dout;(N*T;M)]]; reshapeM[x;(N*T;D)]];
    db:2 sum/dout;
    `dx`dw`db!(dx;dw;db)
 };












