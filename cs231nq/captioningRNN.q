/ util for decoding captions, in coco data it should be 2
/ (.i.e. word_to_idx@`$"<END>"), uses global vars from load_coco_data.q
decodeCaptions:{" " sv idx_to_word (1+x?endId)#x}

captioningRNN.init:{[d]
    defaults:(!). flip 
    (
      `dimInput,512;
      `dimWordVec,128;
      `dimHidden,128;
      `useBatchNorm,0b;
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
    dimMul:(`lstm`rnn!4 1)d`cellType;
    d[`wx]:rad[dimWordVec,dimMul*dimHidden]%sqrt dimWordVec;
    d[`wh]:rad[dimHidden,dimMul*dimHidden]%dimHidden;
    d[`b]:(dimMul*dimHidden)#0f;

    / init output to vocab weights
    d[`wVocab]:rad[dimHidden,vocabSize]%sqrt dimHidden;
    d[`bVocab]:vocabSize#0f;

    d
 };


captioningRNN.params:{[d] `wEmbed`wProj`bProj`wx`wh`b`wVocab`bVocab};

/ input d with key:
/   `features: input image features, shape (N;D)
/   `captions: ground-truth captions, integer array shape (N;T) where
/              each element in the range 0<=y[i, t] < V
captioningRNN.loss:{[d]
    captionsIn:-1 _/:d`captions;
    captionsOut:1 _/:d`captions;
    mask:not captionsOut=d`null;
    
    / ################ Forward pass ################
    h0:dot[d`features;d`wProj]+\:d`bProj;

    xCacheEmbedding: wordEmbeddingForward[captionsIn;d`wEmbed];
    x:xCacheEmbedding 0;
    cacheEmbedding:xCacheEmbedding 1;

    hCacheRnn:(`rnn`lstm!rnnForward,lstmForward)[d`cellType]`x`h0`wx`wh`b!(x;h0;d`wx;d`wh;d`b);
    h:hCacheRnn 0;
    cacheRnn:hCacheRnn 1;
    scoresCacheScores:temporalAffineForward[h;d`wVocab;d`bVocab];
    scores:scoresCacheScores 0;
    cacheScores:scoresCacheScores 1;

    lossDscores:temporalSoftmaxLoss[scores;captionsOut;mask];
    loss:lossDscores 0;
    dscores:lossDscores 1;

    / ################ Backward pass ###############
     
    dxDwDb:temporalAffineBackward[dscores;cacheScores];
    grads:(`rnn`lstm!rnnBackward,lstmBackward)[d`cellType][dxDwDb`dx;cacheRnn];

    dwEmbed:wordEmbeddingBackward[grads`dx;cacheEmbedding];

    dwProj:dot[flip d`features;grads`dh0];
    dbProj:sum grads`dh0;
    
    gradRes:`wProj`bProj`wEmbed`wx`wh`b`wVocab`bVocab!
             (dwProj;dbProj;dwEmbed;grads`dwx;grads`dwh;grads`db;dxDwDb`dw;dxDwDb`db);
    
    (loss;gradRes)
 };

captioningRNN.sample:{[d]
    features:d`features;
    N:count features;
    maxLength:dget[d;`maxLength;30];
    captions:d[`null]*(N;maxLength)#1j;
    
    h0:dot[features;d`wProj]+\:d`bProj;
    prevH:h0;
    prevC:h0*0f;
    
    / current word (start word)
    capt:d[`start]*(N;1)#1j;
    d[`capt`prevH`prevC`captions]:(capt;prevH;prevC;());
    
    / d has `capt`captions`prevH
    sampleLoop:{[d;wEmbed;wVocab;bVocab;wx;wh;b]
        wordEmbed:first wordEmbeddingForward[d`capt;wEmbed];
        hc:$[`rnn~d`cellType;
               rnnStepForward[squeeze wordEmbed;d`prevH;wx;wh;b];
               lstmStepForward`x`prevH`prevC`wx`wh`b!(squeeze wordEmbed;d`prevH;d`prevC;wx;wh;b)
            ];
        h:hc 0;
        c:hc 1;
        
        scores:first temporalAffineForward[enlist each h;wVocab;bVocab];
        / TODO: confirm this is correct
        idxBest:squeeze {x?max x}''[scores];
        d[`captions],:enlist idxBest;

        d[`prevH]:h;
        if[`lstm~d`cellType;d[`prevC]:c];
        d[`capt]:idxBest;
        d
     };

    res:maxLength sampleLoop[;d`wEmbed;d`wVocab;d`bVocab;d`wx;d`wh;d`b]/d;
    flip res`captions
 };

sampleCocoMinibatch:{[d;split;batchSize] d,captioningSolver.genBatch d:@[d;`split`batchSize;:;(split;batchSize)]}

/ inputs:
/   res - result of training, i.e. captioningRNN.train ....
/   smallData - small data to run on 
/   split - `test or `val
sampleCaptions:{[smallData;res;split]
    lg "########## Running captions on ",string[split]," data ###########";
    minibatch:sampleCocoMinibatch[smallData;split;2];
    gtCaptions:minibatch`captions;
    features:minibatch`imageFeatures;
    captionTrainRes:captioningRNN.sample @[res;`features;:;features];
    {[gtRes;res;url] lg "for url: \n",url,"\nground truth captions are: \n",(decodeCaptions gtRes),"\n";lg "train res captions are: \n",(decodeCaptions res),"\n";}./: flip (minibatch`captions;captionTrainRes;minibatch`urls);
 };










