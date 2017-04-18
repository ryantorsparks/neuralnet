h:0.00001;

evalNumericalGradient:{[f;input]
    fx:f input;
    grad:0*fx;
    fxPlusH:f input+h;
    fxMinusH:f input-h;
    grad:(sum/)(fxPlusH-fxMinusH)%2*h;
    grad
 };


numGradInnerFunc:{[f;d;param;ind;plusMinus]
    index:$[99h=type d;param,ind;(),ind];
    f .[d;index;plusMinus;h]
 };

numGradOneIndChange:{[f;d;param;ind]
    proj:numGradInnerFunc[f;d;param;ind;];
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

