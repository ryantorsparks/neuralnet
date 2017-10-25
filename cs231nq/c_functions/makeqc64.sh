echo "making .so file for $1"
gcc -bundle -D KXVER=3 -m64 -Ofast -undefined dynamic_lookup $1.c -o /Users/ryansparks/qod/m64/$1.so
