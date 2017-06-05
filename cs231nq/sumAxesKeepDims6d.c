#include"k.h"
#include <stdio.h>
// args are (m [6 dim matrix]; res [6 dim matrix of 0s]; mShape [list length 6]; axis (int, where to sum/collapse); )
K sumAxesKeepDims6d(K m, K res, K mShape, K kaxis){
    // m has shape (A;B;C;D;E;F), extract into vars
    long shape[m->n];
    long axis = kaxis->j;
    int i;
    for(i=0;i<mShape->n;++i) shape[i]=(long)kJ(mShape)[i];
    long A, B, C, D, E, G, AA;
    long a, b, c, d, e, g, aa;
    AA=shape[axis];
    shape[axis]=1;
    A = shape[0];
    B = shape[1];
    C = shape[2];
    D = shape[3];
    E = shape[4];
    G = shape[5];
    K il=ktn(KI,6);
    kI(il)[0]=A;
    kI(il)[1]=B;
    kI(il)[2]=C;
    kI(il)[3]=D;
    kI(il)[4]=E;
    kI(il)[5]=G;
    // modify arg2's elements, using indexing into m
    for(a=0;a<A;++a){
        for(b=0;b<B;++b){
            for(c=0;c<C;++c){
                for(d=0;d<D;++d){
                    for(e=0;e<E;++e){
                        for(g=0;g<G;++g){
                            for(aa=0;aa<AA;++aa){
                                kI(il)[0]=a;
                                kI(il)[1]=b;
                                kI(il)[2]=c;
                                kI(il)[3]=d;
                                kI(il)[4]=e;
                                kI(il)[5]=g;
                                kI(il)[axis]=aa;
                                kF(kK(kK(kK(kK(kK(res)[a])[b])[c])[d])[e])[g]+=kF(kK(kK(kK(kK(kK(m)[kI(il)[0]])[kI(il)[1]])[kI(il)[2]])[kI(il)[3]])[kI(il)[4]])[kI(il)[5]];
                            }
                        }
                    }
                }
            }
        }
    }
    r0(il);
    R r1(res);
}
