/ ########### optimiser (update rule) funcs #############

/ vanilla socastic gradient descent
sgd:{[w;dw;config]
    / config expects `learnRate
    (w-dw*config`learnRate;config)
 };

/ sgd with momentum
/ config is :
/   learnRate - float
/   momentum - foat within (0 1f) giving momentum value, 0 -> plain sgd
/   velocity - float array same shape as w and dw used to store a moving
/              average of the gradients
sgdMomentum:{[w;dw;config]
    / config expects `learnRate`momentum`velocity
    config:where[config~\:(::)] _ config;
    defaults:`learnRate`momentum!0.01 0.9;
    config:defaults,config;
    
    / velocity, default to 0's of shape w
    v:$[(`velocity in key config)and not all null razeo (),config`velocity;config`velocity;w*0.0];

    / momentum update
    v: (v*config`momentum)-dw*config`learnRate;
    config:config,enlist[`velocity]!enlist v;
    (w+v;config)
 };

/ rmsProp update rule, uses mavg of square gradient rules set adaptive per-
/ parameter learning rates
/ config format should be:
/   learnRate - float
/   updateDecayRate - float between 0 and 1f, decay rate of the squared gradient cache
/   epsilon - small float, used for smoothing to avoid dividing by 0
/   cache - mavg of second moments of gradients
rmsProp:{[x;dx;config]
    / d optionals `learnRate`updateDecayRate`epsilon`cache (defaults provided)
    defaults: `learnRate`updateDecayRate`epsilon`cache!(0.01;0.99;1e-8;x*0.0);

    / remove the null initialized ones (replace with default)
    config:where[config~\:(::)] _ config;
    config:defaults,config;
    
    / store next value of x as nextX
    cache:config`cache;
    updateDecayRate:config`updateDecayRate;
    epsilon:config`epsilon;
    learnRate:config`learnRate;
    cache:(cache*updateDecayRate)+(1-updateDecayRate)*dx*dx;
    nextX:x-learnRate*dx%epsilon+sqrt cache;
    config[`cache]:cache;
    (nextX;config)
 };

/ adam update, which incorporates moving averages of both the gradient
/ and its square, and a bias correction term
/ config format should be:
/   learnRate - float
/   beta1 - decay rate for mavg of first moment of gradient
/   beta2 - decay rate for mavg of second moment of gradient
/   epsilon - small float for smoothing to avoid dividing by 0
/   m - moving avg of gradient
/   v - moving average of squared gradient
/   t - iteration number
/ note with this, we flatten matrixes x and dx as doing operations on these
/ and then reshaping at the end is up to 30% faster
adam:{[x;dx;config]
    / d optionals `learnRate`beta1`beta2`epsilon`m`v`t
    shapex:shape x;
    cntx:prd shapex;
    x:razeo x;
    dx:razeo dx;
    defaults:(!) . flip ((`learnRate;1e-3);(`beta1;0.9);(`beta2;0.999);
             (`epsilon;1e-8);(`m;cntx#0f);(`v;cntx#0f);(`t;0));

    / remove the null initialized ones (replace with default)
    config:where[config~\:(::)] _ config;
    config:defaults,config;
    learnRate:config`learnRate;
    beta1:config`beta1;
    beta2:config`beta2;
    epsilon:config`epsilon;
    m:config`m;
    v:config`v;
    t:1+config`t;
    m:(beta1*m)+dx*1-beta1;
    v:(beta2*v)+(1-beta2)*dx*dx;

    / bias correction
    mb:m%1-beta1 xexp t;
    vb:v%1-beta2 xexp t;
    nextX:x-learnRate* mb % epsilon+sqrt vb;
    config[`m`v`t]:(m;v;t);
    (shapex#nextX;config)
 };
