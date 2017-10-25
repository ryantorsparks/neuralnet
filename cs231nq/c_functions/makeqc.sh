echo "making .so file for $1"
gcc -bundle -D KXVER=3 -m32 -Ofast -undefined dynamic_lookup $1.c -o /Users/ryansparks/q/m32/$1.so
