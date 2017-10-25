#include"k.h"
#include <stdio.h>
// args are (dx_cols [6 dim matrix]; xpad [4 dim matrix]; arg1shape (list of 6 longs, shape of arg1);pad (long); stride (long))
K col2im6dInner(K arg1, K arg2, K arg1shape, K padsize, K stridesize){
    // arg1 has shape (C;HH;WW;N;H;W), extract into vars
    long shape[arg1->n];
    int i;
    for(i=0;i<arg1shape->n;++i) shape[i]=(long)kJ(arg1shape)[i];
    int C, HH, WW, N, H, W;
    long c, hh, ww, n, h, w ;
    long pad = padsize->j;
    long stride = stridesize->j;
    C = shape[0];
    HH = shape[1];
    WW = shape[2];
    N = shape[3];
    H = shape[4];
    W = shape[5];
    long out_h, out_w;
    out_h = (H + 2*pad - HH) / stride + 1;
    out_w = (W + 2*pad - WW) / stride + 1;
    printf("C,HH,WW,N,H,W,out_h,out_w,stride, pad are %d %d %d %d %d %d %ld %ld %ld %ld \n",C,HH,WW,N,H,W,out_h,out_w,stride,pad);
    printf(" order of the forloop will be N,C,HH,WW,out_h,out_w -  %d %d %d %d %ld %ld\n",N,C,HH,WW,out_h,out_w);
    // modify arg2's elements, using indexing into arg1
    for(n=0;n<N;++n){
       for(c=0;c<C;++c){
          for(hh=0;hh<HH;++hh){
              for(ww=0;ww<WW;++ww){
                  for(h=0;h<out_h;++h){
                      for(w=0;w<out_w;++w){
                          kF(kK(kK(kK(arg2)[n])[c])[(stride*h+hh)])[stride*w+ww] += kF(kK(kK(kK(kK(kK(arg1)[c])[hh])[ww])[n])[h])[w];
                      }
                  }
              }
          }
      }
    }
    R r1(arg2);
}
