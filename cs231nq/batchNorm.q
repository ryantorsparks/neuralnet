\l nn_util.q

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







