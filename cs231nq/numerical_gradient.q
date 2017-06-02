/ functions for evalutating the (slow) numerical gradients
/ which we use as sanity checks

defaultGradStep:1e-5

getGradStep:{[d]
   $[99h=type d;dget[d;`h;defaultGradStep];defaultGradStep]
 };

evalNumericalGradient:{[f;input]
    fx:f input;
    grad:0*fx;
    / TODO: remove hard coding here
    h:defaultGradStep;
    fxPlusH:f input+h;
    fxMinusH:f input-h;
    grad:(sum/)(fxPlusH-fxMinusH)%2*h;
    grad
 };


numGradInnerFunc:{[f;d;param;ind;plusMinus]
    index:$[99h=type d;param,ind;(),ind];
    h:getGradStep d;
    f .[d;index;plusMinus;h]
 };

numGradOneIndChange:{[f;d;param;ind]
    proj:numGradInnerFunc[f;d;param;ind;];
    h:getGradStep d;
    (proj[+]-proj[-])%2*h
 };

/ e.g
/ {relError[res[1;x];numericalGradient[(first twoLayerNet@);d;x]]}each `w2`b2`w1`b1
numericalGradient:{[f;d;param]
   input:$[99h=type d;d param;d];
   paramShape:shape input;
   inds:cross/[til each paramShape];
   res:numGradOneIndChange[f;d;param]each inds;
   paramShape#res
 };

/ df is array same shape, multiply
numericalGradientArray:{[f;d;df;param]
   input:$[99h=type d;d param;d];
   paramShape:shape input;
   inds:cross/[til each paramShape];
   res:((sum/)df*numGradOneIndChange[f;d;param]@)each inds;
   paramShape#res
 };

/ compare a few numerical gradients
compareNumericalGradients:{[d;reg]
    / d expects `reg`model, any params from model's loss func,
    /           so possibly `x`w1`w2...`b1`b2...
    lg"running numeric gradient check with reg=",string reg;
    d[`reg]:reg;
    lossFunc:value ` sv $[`model in key d;d`model;`twoLayerNet],`loss;
 
    grads:last lossFunc d;
    {[d;grads;lossFunc;param]
        f:(first lossFunc@);
        gradNum:numericalGradient[f;d;param];
        relErr:relError[gradNum;grads param];
        lg"relative error for ",string[param]," is ",-3!relErr;
    }[`model _ d;grads;lossFunc;] each asc {x where not x like "x*"} key grads;
 };


