#include"k.h"
#include <stdio.h>
K expandAxes35Flat6dMatrix(K m, K mShape, K emptyres){
    // m has shape (A;B;C;D;E;F), extract into vars
    long shape[6];
    I i;
    for(i=0;i<6;++i) shape[i]=(long)kJ(mShape)[i];
    long A, B, C, D, E, G;
    long a, b, c, d, e, g;
    // extract the shape for for-loops
    A = shape[0];
    B = shape[1];
    C = shape[2];
    D = shape[3]; // sum along this axis
    E = shape[4];
    G = shape[5]; // sum along this axis
    long emptyIndex;
   for(a=0;a<A;++a){
        for(b=0;b<B;++b){
            for(c=0;c<C;++c){
                for(d=0;d<D;++d){
                    for(e=0;e<E;++e){
                        for(g=0;g<G;++g){
                            // equivalent index in the empty list, to matrix[a;b;c;d;e;g]
                            emptyIndex=a*B*C*D*E*G + b*C*D*E*G + c*D*E*G + d*E*G + e*G + g;
                            kF(emptyres)[emptyIndex]=kF(kK(kK(kK(kK(kK(m)[a])[b])[c])[0])[e])[0];
                        }
                    }
                }
            }
        }
     };
     R r1(emptyres);
}
