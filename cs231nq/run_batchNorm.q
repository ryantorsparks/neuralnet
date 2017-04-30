\l nn_util.q
\l fullyConnected.q
\l numerical_gradient.q
\l softmax.q
\l linear_svm.q
cifarMode:`unflattened
\l load_cifar_data.q

lg "##############################
    Neural nets with batch normalization
    ##############################"

`. upsert `N`D1`D2`D3!200 50 60 3;
x:randArray[N;D1]
w1:randArray[D1;D2]
w2:randArray[D2;D3]
a:dot[(0|dot[x;w1]);w2]
lg "before batch norm, means and dev's of a are "
avg a
dev each flip a

lg "##############################
    Batchnorm forward
    ##############################"

lg "after batch norm (gamma=1, beta=0), mean and dev's are "
aNorm:first batchNormForward[a;D3#1f;D3#0f;enlist[`mode]!enlist`train] 
avg aNorm
dev each flip aNorm

lg "after batch norm (nontrivial gamma, beta)"
aNorm: first batchNormForward[a;1 2 3f;11 12 13f;enlist[`mode]!enlist `train]                                                                                        
avg aNorm
dev each flip aNorm

lg "check test time forward pass by running the training-time forward
    pass many times to warm up the running avgs, and then checking the
    means and variances of activations after a test-time forward pass"
`. upsert `N`D1`D2`D3!200 50 60 3;
w1:randArray[D1;D2]
w2:randArray[D2;D3]
bnParam:(1#`mode)!1#`train
gamma:D3#1f
beta:D3#0f
{[bnParam;x;w1;w2;gamma;beta] last batchNormForward[dot[0f|dot[x;w1];w2];gamma;beta;bnParam]}[;;w1;w2;gamma;beta]/[bnParam;tempx]
res:50{[bnParam;x;w1;w2;gamma;beta] last batchNormForward[dot[0f|dot[x;w1];w2];gamma;beta;bnParam]}[;x;w1;w2;gamma;beta]/bnParam 

x2:randArray[N;D1]
postRes:first batchNormForward[dot[0|dot[x2;w1];w2];gamma;beta;@[res;`mode;:;`test]] 
lg "after batch norm, mean and devs are"
avg postRes
dev each flip postRes

lg "##############################
    Batchnorm backward
    ##############################"


N:4
D:5
x:12+5*randArray[N;D]
gamma:first randArray[1;D]
beta:first randArray[1;D]
dout:randArray[N;D]

bnParam:(!). 1#'`mode`train
lg "get numerical grads"
dxNum:numericalGradientArray[(first batchNormForward[;gamma;beta;bnParam]@);x;dout;`x]
daNum:numericalGradientArray[(first batchNormForward[x;;beta;bnParam]@);gamma;dout;`gamma]
dbNum:numericalGradientArray[(first batchNormForward[x;gamma;;bnParam]@);beta;dout;`beta]

lg "now run batchNorm forward then backward"
cache:@[;1] batchNormForward[x;gamma;beta;bnParam]
dxDaDb: batchNormBackward[dout;cache]

lg "relative errors for dx, dgamma and dbeta are"
relError'[(dxNum;daNum;dbNum);dxDaDb]

lg "Alternative batchnorm backwards using derivs"
N:100
D:500
x:12+5*randArray[N;D]
gamma:first randArray[1;D]
beta:first randArray[1;D]
dout:randArray[N;D]

bnParam:(!). 1#'`mode`train
outCache:batchNormForward[x;gamma;beta;bnParam]

lg "run both batchnorm backwards versions 100 times and compare results/timings"
t1:system"t:100 res1:batchNormBackward[dout;outCache 1]"
t2:system"t:100 res2:batchNormBackwardAlt[dout;outCache 1]"
lg"dx, dgamma, dbeta relative differences are"
relError'[res1;res2]
lg"(time batchNormBackward)%time batchNormBackwardAlt was ",string t1%t2
