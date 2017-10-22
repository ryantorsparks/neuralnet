#include"k.h"
#include <stdio.h>
#include <math.h>
// function to do transpose on a flattened version of a matrix,
// i.e. if m:5 11#55?1f;
//      flipFlat[raze m;5;11;55#0f]~raze flip m
// example (b (list length 100); m (200*100 list); nrows 200; ncolumns 100)
K flipFlat(K flatm, K nrows, K ncolumns, K emptyres){
    I i,j,rows,columns,r,c,index1,index2;
    F resvalue;
    rows=nrows->n;
    columns=ncolumns->n;
    for(r=0;r<rows;++r){
        for(c=0;c<columns;++c){
            index1=r*columns+c;
            index2=c*rows+r;
            resvalue=kF(flatm)[index1];
            kF(emptyres)[index2]=resvalue;
        }
     };
     R r1(emptyres);
}
