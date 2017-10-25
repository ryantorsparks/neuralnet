#include"k.h"
#include <stdio.h>
// example (b (list length 100); m (200*100 list); nrows 200; ncolumns 100)
K addDotBias(K b, K m, K nrows, K ncolumns){
    I i,j,r,c;
    r=nrows->n;
    c=ncolumns->n;
    for(i=0;i<c;++i){
        for(j=0;j<r;++j){
            kF(m)[(i*r)+j]+=kF(b)[i];
        }
     };
     R r1(m);
}
