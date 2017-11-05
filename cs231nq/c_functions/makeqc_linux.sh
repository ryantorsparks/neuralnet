echo "making .so file for $1"
gcc -shared  -DKXVER=3 -O2 $1.c -o $QHOME/l32/$1.so -fPIC
