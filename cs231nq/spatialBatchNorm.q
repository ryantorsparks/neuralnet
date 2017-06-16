/ spatial batch normalization funcs, for convnets

/ computes forward pass for spatial batch norm
/ inputs are: 
/   x - input data, shape (N;C;H;W)
/   gamma - scale param, list length C
/   beta - shift param, list length C
/   bnParam - dictionary with:
/     mode - `train or `test
/     eps - constant for numeric stability
/     momentum -constant for runnin mean/variance
/     runningMean - list length D, running mean of features
/     runningVar - list length D, running variance of features
/ returns, (out;cache), where out is shape (N;C;H;W)
spatialBatchNormForward:{[x;gamma;beta;bnParam]
    xShape:shape x;
    N:xShape 0;
    C:xShape 1;
    H:xShape 2;
    W:xShape 3;
    mode:bnParam`mode;
    eps:dget[bnParam;`eps;1e-5];
    momentum:dget[bnParam;`momentum;0.9];
    runningMean:dget[bnParam;`runningMean;C#0f];
    runningVar:dget[bnParam;`runningVar;C#0f];
    cShape:1,C,1 1;

    / train mode
//    if[mode=`train;
        / means for each channel
        prdNHW:prd N,H,W;
        mu:reshapeM[sumAxes[x;0 2 3]*1%prdNHW;cShape];

        / doing "x-mu" but extrapolating/expanding mu to x's dimensions
        xcorrected:x-first[broadcastArrays[mu;x]];
        variance:reshapeM[sumAxes[xcorrected xexp 2;0 2 3]*1%prdNHW;cShape];

        / expand eps+variance to same dimension as xcorrected, for division
        xhat:xcorrected%last broadcastArrays[xcorrected;sqrt eps+variance];

        / turn lists gamma&beta into arrays same shape as xhat
        gammaExpanded:first[broadcastArrays[reshapeM[gamma;cShape];xhat]];
        betaExpanded:first broadcastArrays[reshapeM[beta;cShape];xhat];
        out:betaExpanded+xhat*gammaExpanded;

        / update running var/means
        runningMean:(momentum*runningMean)+razeo[mu]*1-momentum;
        runningVar:(momentum*runningVar)+razeo[variance]*1-momentum;
        
        / cache for back pass
        cache:`mu`variance`x`xhat`gamma`beta`bnParam!(mu;variance;x;xhat;gamma;beta;bnParam);
//       ];

    / store the updated running means back into bnParam
    bnParam:bnParam,`runningMean`runningVar!(runningMean;runningVar);
    (out;cache;bnParam) 
 };



