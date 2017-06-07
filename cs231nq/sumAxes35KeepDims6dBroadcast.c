#include"k.h"
#include <stdio.h>
// args are (m [6 dim matrix]; mShape [list length 6] )
// will sum along axes 3 and 5 (4th and 6th dim)
K sumAxes35KeepDims6dBroadcast(K m, K mShape){
    // m has shape (A;B;C;D;E;F), extract into vars
    long shape[m->n];
    int i;
    for(i=0;i<mShape->n;++i) shape[i]=(long)kJ(mShape)[i];
    long A, B, C, D, E, G;
    long a, b, c, d, e, g;
    // extract the shape for for-loops
    A = shape[0];
    B = shape[1];
    C = shape[2];
    D = shape[3]; // sum along this axis
    E = shape[4];
    G = shape[5]; // sum along this axis
    // keep running total
    double total=0;
    // sum and expand along 4th and 6th dimensions
    for(a=0;a<A;++a){
        for(b=0;b<B;++b){
            for(c=0;c<C;++c){
                for(e=0;e<E;++e){
                    total=0;
                    for(d=0;d<D;++d){
                        for(g=0;g<G;++g){
                            total+=kF(kK(kK(kK(kK(kK(m)[a])[b])[c])[d])[e])[g];
                        }
                    }
                    for(d=0;d<D;++d){
                       for(g=0;g<G;++g){
                           kF(kK(kK(kK(kK(kK(m)[a])[b])[c])[d])[e])[g]=total;
                       }
                    }
                }
            }
        }
     };
     R r1(m);
}
