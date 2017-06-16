/ batch normalisation funcs (and spatial batchnorm)

/ forward pass for batch normalization
/ inputs: 
/   x - data of shape (N;D)
/   gamma - scale paramete shape, list length D
/   beta - shift parameter, list length D
/   bnParam - dictionary with:
/     mode - `train or `test
/     eps - constant for numeric stability
/     momentum -constant for runnin mean/variance
/     runningMean - list length D, running mean of features
/     runningVar - list length D, running variance of features
/ returns, (out;cache;bnParam), where out is shape (N;D)
/ TODO: change cache output to be dict
batchNormForward:{[x;gamma;beta;bnParam]
    mode:bnParam`mode;
    eps:dget[bnParam;`eps;1e-5];
    momentum:dget[bnParam;`momentum;0.9];
    shapex:shape x;
    N:shapex 0;
    D:shapex 1;
    runningMean:dget[bnParam;`runningMean;D#0f];
    runningVar:dget[bnParam;`runningVar;D#0f];
    if[not mode in `train`test;'"batchNormForward: mode must be in `train`test"];
    
    / store updated running means 
//    if[mode=`train;
    if[1b;
        mu:avg x;
        xcorrected:x-\:mu;
        variance:avg xcorrected xexp 2;
        std:sqrt variance+eps;
//        show"avg, var and std are ",-3!2 sublist '(mu;variance;std);
        xhat:xcorrected%\:std;
        out:beta+/:gamma*/:xhat;
        cache:(mode;x;gamma;xcorrected;std;xhat;out;variance+eps);
        / update running avg of mean
        runningMean*:momentum;
        runningMean+:mu*1-momentum;
        runningVar*:momentum;
        runningVar+:variance*1-momentum;
      ];
//    if[mode=`test;
    if[0b;
        std:sqrt runningVar+eps;
        xhat:(x-\:runningMean)%\:std;
        out:beta+/:gamma*/:xhat;
//      show"test:avg, var and std are ",-3!2 sublist '(avg x;runningVar+eps;std);
        cache:(mode;x;xhat;gamma;beta;std);
      ];
    
    / store the updated running means back into bnParam
    bnParam:bnParam,`runningMean`runningVar!(runningMean;runningVar);
    (out;cache;bnParam)
 };

/ backward batchnorm
/ inputs are:
/   dout: upstream derivs, shape (N;D)
/   cache: variable of intermediates from batchNormForward
/ returns (dx;dgamma;dbeta):
/   dx - grad wrt. inputs x, shape (N;D)
/   dgamma - grad wrt. scalar param gamma, list length D
/   dbeta - grad wrt. shift param beta, list length D
batchNormBackward:{[dout;cache]
    mode:cache 0;
    if[mode=`train;
        x:cache 1;
        gamma:cache 2;
        xc:cache 3;
        std: cache 4; 
        xn:cache 5;
        out:cache 6;
        N:count x;
        dbeta:sum dout;
        dgamma:sum xn*dout;
        dxn:gamma*/:dout;
        dxc:dxn%\:std;
        dstd:neg sum (dxn*xc)%\:std*std;
        dvar:.5*dstd%std;
        dxc+:dvar*/:xc*2%N;
        dmu:sum dxc;
        dx:dxc-\:dmu%N;
      ];
    if[mode=`test;
        x:cache 1;
        xn:cache 2;
        gamma:cache 3;
        beta:cache 4; 
        std:cache 5;
        dbeta:sum dout;
        dgamma:sum xn*dout;
        dxn:gamma*dout;
        dx:dxn%std;
      ];
    `dx`dgamma`dbeta!(dx;dgamma;dbeta)
 };

/ batchnorm using differentiation, alternate method,
/ see http://costapt.github.io/2016/07/09/batch-norm-alt/
/ cache comes from train section of batchNormForward
/ i.e.  cache:(mode;x;gamma;xcorrected;std;xhat;out;variance+eps);
/ xhat is (x-\:mu)%sqrt variance+eps
/ variance is avg (x-\:mu) xexp 2
/ TODO: change output of bath norm forward to use dict
batchNormBackwardAlt:{[dout;cache]
    gamma:cache 2;
    xcorrected:cache 3;
    std:cache 4;
    xhat:cache 5;
    ivarEps:1%cache 7;
    N:count dout; 
    dbeta:sum dout;
    dgamma:sum xhat*dout;
    dx:(gamma%std*N)*/:((N*dout)-\:dbeta)-xcorrected*\:ivarEps*sum dout*xcorrected;
    `dx`dgamma`dbeta!(dx;dgamma;dbeta)
 };

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
        xcorrected:broadcastArraysFunc[x;mu;-];
        variance:reshapeM[sumAxes[xcorrected xexp 2;0 2 3]*1%prdNHW;cShape];

        / expand eps+variance to same dimension as xcorrected, for division
        xhat:broadcastArraysFunc[xcorrected;sqrt eps+variance;%];

        / turn lists gamma&beta into arrays same shape as xhat, then add and multiply
        / i.e equivalent to (xhat*gamma)+beta
        out:broadcastArraysFunc[broadcastArraysFunc[xhat;reshapeM[gamma;cShape];*];reshapeM[beta;cShape];+];

        / update running var/means
        runningMean:(momentum*runningMean)+razeo[mu]*1-momentum;
        runningVar:(momentum*runningVar)+razeo[variance]*1-momentum;
        
        / cache for back pass
        cache:`xcorrected`mu`variance`x`xhat`gamma`beta`bnParam!(xcorrected;mu;variance;x;xhat;gamma;beta;bnParam);
//       ];

    / store the updated running means back into bnParam
    bnParam:bnParam,`runningMean`runningVar!(runningMean;runningVar);
    (out;cache;bnParam) 
 };


/ spatial batchnorm backward pass
/ inputs are:
/   dout: upstream derivs, shape (N;C;H;W)
/   cache: variable of intermediates from spatialBatchNormForward
/ returns (dx;dgamma;dbeta):
/   dx - grad wrt. inputs x, shape (N;C;H;W)
/   dgamma - grad wrt. scalar param gamma, list length C
/   dbeta - grad wrt. shift param beta, list length C
spatialBatchNormBackward:{[dout;cache]
    mu:cache`mu;
    variance:cache`variance;
    x:cache`x;
    xShape:shape x;
    N:xShape 0;
    C:xShape 1;
    H:xShape 2;
    W:xShape 3;
    cShape:1,C,1 1;
    xhat:cache`xhat;
    xcorrected:cache`xcorrected;
    gamma:reshapeM[cache`gamma;cShape];
    beta:reshapeM[cache`beta;cShape];
    bnParam:cache`bnParam;
    mode:bnParam`mode;
    eps:dget[bnParam;`eps;1e-5];

    dbeta:sumAxes[dout;0 2 3];
    dgamma:sumAxes[dout*xhat;0 2 3];
    
    prdNHW:prd N,H,W;

    / caclulate dx (bam/baa = broadcastArrayMultiply/Add))
    dx:(1%prdNHW)*bam[;gamma] bam[;reciprocal[sqrt variance+eps]]@
        (prdNHW*dout)-baa[;reshapeM[sumAxes[dout;0 2 3];cShape]]@ bam[xcorrected;(1%variance+eps)*reshapeM[sumAxes[dout*xcorrected;0 2 3];cShape]];
    `dx`dgamma`dbeta!(dx;dgamma;dbeta)
 };











