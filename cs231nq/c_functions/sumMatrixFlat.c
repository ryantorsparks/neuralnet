#include"k.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
// function to do the flat version of sum matrix
// i.e. if m:5 11#55?1f;
//      sumMatrixFlat[raze m;5;11]~sum m
// example (b (list length 100); m (200*100 list); nrows 200; ncolumns 100)
K sumMatrixFlat(K flatm, K nrows, K ncolumns){
    I i,j,rows,columns,r,c;
    rows=nrows->n;
    columns=ncolumns->n;
    K emptyres = ktn(KF,columns);
    for(r=0;r<rows;++r){
        for(c=0;c<columns;++c){
            kF(emptyres)[c]+=kF(flatm)[r*columns+c];
        }
     };
     R(emptyres);
}
