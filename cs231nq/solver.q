/ ########### .solver.class functions ###########

/ optional args:
/   `updateRule - e.g `sgd
/   `optimConfig - dict of hyperparams, each update rule needs different 
/       hyperparams, but all rules require a learnRate param
/   `learnRateDecay - after each epoch learnRate is multiplied by this
/   `batchSize - minibatch size for training
/   `numEpochs - number of epochs to run during training
/   `printEvery - training losses will be printery every printEvery iterations
.solver.init:{[d]
    / d expects `model (getModelValue)
    d:nulld,d;
   
    / add on initial default params for the model
    d:getModelValue[d;`init];
    defaults:(!) . flip (
        (`cnt;0);
        (`updateRule;`sgd);
        (`optimConfig;()!());
        (`optimConfigHistory;::);
        (`learnRateDecay;1.0);
        (`batchSize;100);
        (`numEpochs;10);
        (`printEvery;10)
        );
    d:defaults,d;
    if[not count key dur:d`updateRule;'"update rule `",string[dur]," not defined"];
    d
 };

/ reset a bunch of dict variables
.solver.reset:{[d]
    / d expects `optimConfig`model
    / book-keeping variables
    optimd:d`optimConfig;

    / use [model].params[d] to get param list
    modelParams:getModelValue[d;`params];
    if[d`useBatchNorm;modelParams,:raze d`betaParams`gammaParams];
    d,:(!) . flip (
        (`epoch;0);
        (`bestValAcc;0.0);
        (`bestParams;());
        (`lossHistory;());
        (`trainAccHistory;());
        (`valAccHistory;());
        / store optimConfig for each param in d
        (`optimConfigs;([]p:modelParams)!count[modelParams]#enlist optimd)
        );
    d
 };


.solver.genBatch:{[d]
    numTrain:count d`xTrain;
    batchMask:neg[d`batchSize]?numTrain;
    xBatch:d[`xTrain] batchMask;
    yBatch:d[`yTrain] batchMask;
    (xBatch;yBatch)
 };

/ allow for xTrain to be either read from mapped array, or already an in-mem blob
.solver.xTrainParser:{[x]
    $[3072=count x 0;3 32 32#/:x;x]
 };
.flat.solver.xTrainParser:(::)

/ step function
.solver.step:{[d]
    / d expects `xTrain`yTrain`batchSize`lossHistory`optimConfigs`updateRule
    / possibly (???( 
    / create mini batch
    /numTrain:count d`xTrain;
    /batchMask:neg[d`batchSize]?numTrain;
    /xBatch:d[`xTrain] batchMask;
    /yBatch:d[`yTrain] batchMask;
//    batches:getClassValue[`.solver.genBatch;dget[d;`class;`]]d;
    batches:.solver.genBatch[d];
    xBatch:.solver.xTrainParser batches 0;
    / do data augmentation, mirror half the images randomly
    if[1b~d`dataAugmentation;
        xBatch:@[xBatch;{neg[x div 2]?x}d`batchSize;reverse each]];
    yBatch:batches 1;
    .solver.i.step[d;xBatch;yBatch]
 };

.solver.i.step:{[d;xBatch;yBatch]
    / compute loss and grad of mini batch
    modelParams:getModelValue[d;`params];
  
    / ??? about the stuff after `reg
    lossGradKeys:modelParams,`reg`dropoutParam`useBatchNorm`bnParams`wParams`dwParams`bParams`dbParams`betaParams`dbetaParams`gammaParams`dgammaParams`layerInds`model`filterSize`L`M`dropout`useDropout`null`cellType`flat;
    if[d`useBatchNorm;lossGradKeys,:`gammaParams`betaParams,getModelValue[d;`bnParams]];
    inputDict:$[d[`class]~`captioning;`features`captions!(xBatch;yBatch);`x`y!(xBatch;yBatch)];
    lossGrad:getModelValue[ (inter[lossGradKeys;key d]#d),inputDict;`loss];
    loss:lossGrad 0;
    grads:lossGrad 1;
    if[null loss;break];
    d[`lossHistory],:loss;

    / parameter update
    updateParams:modelParams,(();raze d`gammaParams`betaParams)d`useBatchNorm;
    dchange:updateParams#d;
    d:{[d;p;w;grads] 
        dw:grads p;
        config:d[`optimConfigs]p;
        nextWConfig:d[`updateRule][w;dw;config];
        nextW:nextWConfig 0;
        nextConfig:nextWConfig 1;
        d[p]:nextW;
        optimConfigs:d`optimConfigs;
        if[count missingConfigParams:key[nextConfig] except cols optimConfigs;
            d[`optimConfigs]:![optimConfigs;();0b;missingConfigParams!count[missingConfigParams]#enlist (::)];
          ];
        d[`optimConfigs;p]:nextConfig;
        d
    }[;;;grads]/[d;key dchange;value dchange];
    d
 };

/ accuracy check
/ d is `x`y`numSamples`batchSize
.solver.checkAccuracy:{[d]
    / d expects `x`y`model`numSamples`batchSize
    / possibly sumbsample the data
    N:count d`x;
//    batchSize:d`batchSize;
    batchSize:100;
    x:d`x;
    y:d`y;
    if[(not null numSamples)&N>numSamples:d`numSamples;
        mask:neg[numSamples]?N;
        N:numSamples;
        x@:mask;
        y@:mask;
      ];
    x:.solver.xTrainParser x;
    numBatches:N div batchSize;

    / make sure we do at least enough batches
    if[not 0=N mod batchSize;
        numBatches+:1];

    / get indices of the individual batches, no overlaps
    inds:(batchSize*til numBatches)_til N;

    / get loss func, as it only has x, no y, should just return loss)
    lossFunc:` sv `,d[`model],`loss;

    / also get index of each max entry in resulting loss array
    yPred:raze getMaxIndex[lossFunc;`y _ d] peach x inds;

    / finally, return accuracy
    avg yPred=y
 };

// helper func
getMaxIndex:{[f;d;x]{x?max x}each f @[d;`x;:;x]};
.flat.getMaxIndex:{[f;d;x]{x?max x}each (#). f @[d;`x;:;x]};

/ train function
.solver.train:{[d]
    / d expects `xTrain`batchSize`numEpochs
    / first initialize d (this will first call model specific d[`model].init func, then
    / fill in blanks with default values)
    d: .solver.init d;

    / then reset everything
    d: .solver.reset d;

    / get # of trainings, numEpochs, iters per epoch etc.
    numTrain:count d`xTrain;
    iterationsPerEpoch:1|numTrain div d`batchSize;
    numIterations:d[`numEpochs]*iterationsPerEpoch;
    d[`numIterations`iterationsPerEpoch]:numIterations,iterationsPerEpoch;
    res:numIterations .solver.i.train/d;
    
    / finally, swap in best params (only the model params though, leave
    / the rest in tact (e.g. loss/accuracy histories)
    res:res,getModelValue[res;`params]#res`bestParams;
    res
 };

/ called iteratively by sover.train
.solver.i.train:{[d]
    / d expects `cnt`numIterations`printEvery`lossHistory`iterationsPerEpoch`epoch
    /           `optimConfigs`learnRate`learnRateDecay`model`batchSize`xTrain`yTrain
    /           `trainAccHistory`valAccHistory`bestValAcc
    / possibly print training loss
    cnt:d`cnt;
    numIterations:d`numIterations;

    / step
    d:.solver.step d;

    if[0=cnt mod d`printEvery;
        lgts"Iteration: ",string[d`cnt],"/",string[numIterations]," loss: ",string last d`lossHistory;
      ];

    / at end of every epoch, increment epoch counter, decay learnRate
    if[epochEnd:0=(1+cnt)mod d`iterationsPerEpoch;
        d[`epoch]+:1;
        d[`optimConfigs;;`learnRate]*:d`learnRateDecay;
      ];

    / check training and validation accuracy on first+last iteration,
    / and at the end of every epoch
    modelParams:getModelValue[d;`params];
    if[any (cnt=0;cnt=numIterations+1;epochEnd);
        lg"checking accuracies";
        checkKeys:modelParams,`bParams`wParams`layerInds`useBatchNorm`filterSize`L`M`reg`dropout`useDropout`dropoutParam`flat;
        if[d`useBatchNorm;checkKeys,:`bnParams`betaParams`gammaParams,getModelValue[d;`bnParams]];
        trainAcc:.solver.checkAccuracy (inter[checkKeys;key d]#d),`model`x`y`batchSize`numSamples`useDropout!d[`model`xTrain`yTrain`batchSize],1000,0b;
        valAcc:.solver.checkAccuracy (inter[checkKeys;key d]#d),`model`x`y`batchSize`numSamples`useDropout!d[`model`xVal`yVal`batchSize],0N,0b;
        d[`trainAccHistory],:trainAcc;
        d[`valAccHistory],:valAcc;
        lgts"Epoch: ",string[d`epoch],"/",string[d`numEpochs]," train acc: ",string[trainAcc]," val acc: ",string[valAcc];
        
        / keep track of the best model
        if[valAcc>d`bestValAcc;
            d[`bestValAcc]:valAcc;
            d[`bestParams]:getModelValue[d;`params]#d;
          ];
      ];
    d[`cnt]+:1;
    d
 };


/ ################ captioning .solver.specific funcs ################
captioningSolver.init:{[d]
    / d expects `model (getModelValue)
    d:nulld,d;
   
    / add on initial default params for the model
    d:getModelValue[d;`init];
    defaults:(!). flip (
        `updateRule`sgd;
        `cnt,0;
        (`optimConfig;()!());
        `learnRateDecay,1f;
        `batchSize,100;
        `numEpochs,10;
        `printEvery,10;
        `class`captioning
        );
    d:defaults,d;
    d
 };
        
captioningSolver.genBatch:{[d]
    batchSize:dget[d;`batchSize;100];
    split:string dget[d;`split;`train];
    / either valCaptions or trainCaptions
    captionsData:d@`$split,"Captions";
    batchMask:$[`batchMask in key d;d`batchMask;neg[batchSize]?count captionsData];
    captions:captionsData batchMask;
    imageIdxs:d[`$split,"ImageIdxs"]batchMask;
    imageFeatures:(d@`$split,"Features")@imageIdxs;
    urls:d[`$split,"Urls"]imageIdxs;
    `captions`imageFeatures`urls!(captions;imageFeatures;urls)
 };


captioningSolver.step:{[d]
    d[`split]:`train;
    batches:captioningSolver.genBatch d;
    .solver.i.step[d;;]. batches`imageFeatures`captions
 };

/ train function
captioningSolver.train:{[d]
    / d expects `xTrain`batchSize`numEpochs
    / first initialize d (this will first call model specific d[`model].init func, then
    / fill in blanks with default values)
    d: captioningSolver.init d;

    / then reset everything
    d: .solver.reset d;

    / get # of trainings, numEpochs, iters per epoch etc.
    numTrain:count d`trainCaptions;
    iterationsPerEpoch:1|numTrain div d`batchSize;
    numIterations:d[`numEpochs]*iterationsPerEpoch;
    d[`numIterations`iterationsPerEpoch]:numIterations,iterationsPerEpoch;
    res:numIterations captioningSolver.i.train/d;

    / finally, swap in best params (only the model params though, leave
    / the rest in tact (e.g. loss/accuracy histories)
//    res:res,getModelValue[res;`params]#res`bestParams;
    res
 };
 
captioningSolver.i.train:{[d]
    / d expects `cnt`numIterations`printEvery`lossHistory`iterationsPerEpoch`epoch
    /           `optimConfigs`learnRate`learnRateDecay`model`batchSize`xTrain`yTrain
    /           `trainAccHistory`valAccHistory`bestValAcc
    / possibly print training loss
    cnt:d`cnt;
    numIterations:d`numIterations;

    / step
    d:captioningSolver.step d;

    if[(0=cnt mod d`printEvery)or numIterations=1+cnt;
        lgts"Iteration: ",string[d`cnt],"/",string[numIterations]," loss: ",string last d`lossHistory;
      ];

    / at end of every epoch, increment epoch counter, decay learnRate
    if[0=(1+cnt)mod d`iterationsPerEpoch;
        d[`epoch]+:1;
        d[`optimConfigs;;`learnRate]*:d`learnRateDecay;
      ];
    d[`cnt]+:1;
    d
 };







