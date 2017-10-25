#include"k.h"
#include <stdio.h>
// args are (m [6 dim matrix]; mShape [list length 6]; kaxes (int list, where to sum/collapse); )
// args are (m1 [6 dim matrix, e.g. dims are 50 32 16 1 16 1]; m2 [6 dim matrix, dims are 50 32 16 2 16 2];shape1; shape2)
K maskBroadcast6dAxes35(K m1, K m2, K mShape1, K mShape2){
    // m has shape (A;B;C;D;E;F), extract into vars
    long shape[m2->n];
    int i;
    for(i=0;i<mShape2->n;++i) shape[i]=(long)kJ(mShape2)[i];
    long A, B, C, D, E, G;
    long a, b, c, d, e, g;
    A = shape[0];
    B = shape[1];
    C = shape[2];
    D = shape[3]; // sum and expand this axis for m1
    E = shape[4];
    G = shape[5]; // sum and expand this axis for m1
    double val1, val2, maskval;
    // modify arg2's elements, using indexing into m
    for(a=0;a<A;++a){
        for(b=0;b<B;++b){
            for(c=0;c<C;++c){
                for(d=0;d<D;++d){
                    for(e=0;e<E;++e){
                        for(g=0;g<G;++g){
                            val2=kF(kK(kK(kK(kK(kK(m2)[a])[b])[c])[d])[e])[g];
                            val1=kF(kK(kK(kK(kK(kK(m1)[a])[b])[c])[0])[e])[0];
                            if( val1 == val2){
                                maskval = 1;
                            }
                            else {
                                maskval = 0;
                            }
                            kF(kK(kK(kK(kK(kK(m2)[a])[b])[c])[d])[e])[g]=maskval;
                        }
                    }
                }
            }
        };
     };
     R r1(m2);
}
