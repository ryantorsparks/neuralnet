\l nn_util.q

/ dropout forward pass function
/ inputs:
/   x - input data, of any shape
/   dropout param - dictionary with following:
/       p - dropout parameter, a float, we keep each neuron output with probability p
/       mode - `test or `train, if the mode is `train, do dropout, if `test, just
/              return input
/       seeed - random seed, so we can make this function deterministic for
/               gradient checking
/ outputs - (out;cache):
/   out - an array with same shape as x
/   cache - (dropoutParam;mask), in training mode mask is the dropout
/           mask that was used to multiply th einput, in test mask in null
dropoutForward:{[x;dropoutParam]
    p:dropoutParam`p;
    mode:dropoutParam`mode;
    if[not mode in `train`test;'"mode must be in `train`test, supplied was ",-3!mode];
    if[`seed in key dropoutParam;system"S ",string d`seed];

    / random mask list, used later for cache
    mask:();

    / if training, randomly generate shape of x, check against p
    / otherwise, (if in `test mode), just return x as out 
    out:$[mode=`train;
            [ mask:((prd[shapex:shape x]?1f)<p)%p;
              x*shapex#mask
            ];
            x
         ]; 

    cache:(dropoutParam;mask);
    (out;cache)
 };

/ backward pass function for dropout
/ inputs: 
/   dout - upstream derivatives, any shape
/   cache - (dropoutParam [dict];mask), from dropoutForward
dropoutBackward:{[dout;cache]
    dropoutParam:cache 0;
    mask:cache 1;
    mode:dropoutParam`mode;

    dx:$[mode=`train;
           dout*mask;
           dout
        ];

    dx
 };




