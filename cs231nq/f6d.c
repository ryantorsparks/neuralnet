#include"k.h"
#include <stdio.h>
// args are (dx_cols [6 dim matrix]; xpad [4 dim matrix]; pad; stride)
K f6d(K arg1, K arg2, K arg1shape, K padsize, K stridesize){
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
    K d;
    d=ktn(KJ,8);
    kJ(d)[0]=C;
    kJ(d)[1]=HH;
    kJ(d)[2]=WW;
    kJ(d)[3]=N;
    kJ(d)[4]=H;
    kJ(d)[5]=W;
    kJ(d)[6]=out_h;
    kJ(d)[7]=out_w;
    K arg2n, arg2c, arg2h, arg2hStridehh, arg1c, arg1hh, arg1ww, arg1h;
    for(n=0;n<N;++n){
       arg2n=kK(arg2)[n];
       r1(arg2);
       for(c=0;c<C;++c){
          arg2c=kK(arg2n)[c];
          arg1c=kK(arg1)[c];
          r1(arg1);
          for(hh=0;hh<HH;++hh){
              arg1hh=kK(arg1c)[hh];
              for(ww=0;ww<WW;++ww){
                  arg1ww=kK(arg1hh)[ww];
                  for(h=0;h<out_h;++h){
                      arg2hStridehh=kK(arg2c)[(stride*h+hh)];
                      arg1h=kK(kK(arg1ww)[n])[h];
                      for(w=0;w<out_w;++w){
                          kJ(arg2hStridehh)[stride*w+ww] += kJ(arg1h)[w];
                      }
                  }
              }
          }
      }
    }
//    R r1(arg2);
    R(arg2);
}
