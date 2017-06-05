#include<stdlib.h>
#include<stdio.h>
#include<string.h>

int * intdup(int const * src, size_t len)
{
   int * p = malloc(len * sizeof(int));
   memcpy(p, src, len * sizeof(int));
   return p;
};

int main() {
  int a[2][4] = { 10, 11, 12, 13, 14, 15, 16, 17};
  int * q = intdup(a,8);
  for(int i=0;i<2;i++) {
      for(int j=0;j<4;j++) {
         printf("newlist at %d %d is %d \n",i,j, a[i][j]);
      }
   }
};
