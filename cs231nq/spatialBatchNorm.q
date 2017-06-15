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
/ returns, (out;cach), where out is shape (N;C;H;W)
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
    
    / train mode
    if[mode=`train;
        / means for each channel
        prdNHW:prd N,H,W;
        mu:reshapeM[1%prdNHW*sumAxes[x;0 2 3];1,C,1,1];
        xcorrected:x-first[broadcastArrays[mu;x]];
        variance:reshapeM[1%prdNHW*sumAxes[xcorrected xexp 2;0 2 3];1,C,1,1];
        xhat:xcorrected%

