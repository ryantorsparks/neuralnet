#include"k.h"
#include <stdio.h>
K expandDimFlatMatrix(K flatmatrix, K shape0, K shape1, K shape2, K emptyres){
    I h,i,j;
    for(h=0;h<(shape0->n);++h){
        for(i=0;i<(shape1->n);++i){
           for(j=0;j<(shape2->n);++j){
               kF(emptyres)[h*(shape1->n)*(shape2->n)+i*(shape2->n)+j]=kF(flatmatrix)[h*(shape2->n)+j];
           }
        }
    }
    R r1(emptyres);
}   
