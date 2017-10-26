echo "making .so file for $1"
# note for matrix multiply/cuBLAS stuff, need to also have "-lcurand -lcublas" after "--shared"
nvcc --ptxas-options=-v --compiler-options '-fPIC -DKXVER=3' -o $QHOME/l64/$1.so --shared $1.cu
