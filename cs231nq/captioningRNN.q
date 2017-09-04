nullToken:`$"<NULL>";
startToken:`$"<START>";
endToken:`$"<END>";

captioningRNN.init:{[d]
    defaults:(!). flip 
    (
      `dimInput,512;
      `dimWordVec,128;
      `dimHidden,128;
      `cellType`rnn
    );
    d:defaults,d;
    dimInput:d`dimInput;
    dimHidden:d`dimHidden;
    dimWordVec:d`dimWordVec;
    wordToIdx:d`wordToIdx;
    d[`idxToWord]:(!). (value;key)@\:wordToIdx;
    vocabSize:count wordToIdx;
    d[`null]:wordToIdx nullToken;
    d[`start]:wordToIdx startToken;
    d[`end]:wordToIdx endToken;
    
    / init word vectors
    d[`wEmbed]:0.01*rad vocabSize,d`dimWordVec;
    
    / init CNN -> hidden stat projection params
    d[`wProj]:rad[dimInput,dimHidden]%sqrt dimInput;
    d[`bProj]:dimHidden#0f;

    / init params for the RNN
    dimMul:(`lstm`rnn!1 4)d`cellType;
    d[`wx]:rad[dimWordVec,dimMul*dimHidden]%sqrt dimWordVec;
    d[`wh]:rad[dimHidden,dimMul*dimHidden]%dimHidden;
    d[`b]:(dimMul*dimHidden)#0f;

    / init output to vocab weights
    d[`wVocab]:rad[dimHidden,vocabSize]%sqrt dimHidden;
    d[`bVocab]:vocabSize:0f;

    d
 };


captioningRNN.params:{[d] `wEmbed`wProj`wx`wh`b`wVocab`bVocab};

/ input d with key:
/   `features: input image features, shape (N;D)
/   `captions: ground-truth captions, integer array shape (N;T) where
/              each element in the range 0<=y[i, t] < V
captioningRNN.loss:{[d]
    captionsIn:-1 _/:d`captions;
    captionsOut:1 _/:d`captions;
    mask:captions=nullToken;
    
    / ################ Forward pass ################
    h0:dot[d`features;d`wProj]+/:d`bProj;
    xCacheEmbedding: wordEmbeddingForward[captionsIn;d`wEmbed];
    x:CacheEmbedding 0;
    cacheEmbedding:xCacheEmbedding 1;

    hCacheRnn:(`rnn`lstm!rnnForward,lstmForward)[d`cellType][x;h0;d`wx;d`wh;d`b];
    h:hCacheRnn 0;
    cacheRnn:hCacheRnn 1;

    scoresCacheScores:temporalAffineForward[h;d`wVocab;d`bVocab];
    scores:scoresCacheScores 0;
    cacheScores:scoresCacheScores 1;

    lossDscores:temporalSoftmaxLoss[scores;captionsOut;mask];
    loss:lossDscores 0;
    dscores:lossDscores 1;

    / ################ Backward pass ###############
    dhDwDb:temporalAffineBackward[dscores;cacheScores];
    
    grads:(`rnn`lstm!rnnBackward,lstmBackward)[d`cellType][dhDwDb`dh;cacheRnn];

    dwEmbed:wordEmbeddingBackward[grads`dx;cacheEmbedding];

    dwProj:dot[flip d`features;grads`dh0];
    dbProj:sum grads`dh0;
    






















