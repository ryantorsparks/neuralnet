system"d .debug";

/ debugging funcs, rarely needed (hopefully)
/ will print out time, .Q.w[`used]*1e-9, i.e. memory used in GB
/ @param x - string to log to
memlg:{`..lg@(string[.z.t]," : ",string 1e-9*.Q.w[]`used)," GB - ",x}  

/ takes a given function name, modifies it to print its own name first (along with mem used),
/ used for debugging w-aborts
/ @param x - function name as a symbol
/ e.g:
/   q) f:{100+x*y+z}
/   q) .debug.modifyFunc[`f]
/   q) f[3;4;5]
/     00:20:43.386 : 0.000336688 GB - calling function `f
/   127
modifyFunc:{[funcName]
    if[not 100h=type value funcName;:()];
    (` sv `.old,dropDots funcName) set value funcName;
    funcName set (')[{[f;n;p] memlg "calling function ",-3!n;f . p}[value funcName;funcName];enlist]
 };

/ lifted the following funcs from nick psaris' qtips github 

/ util to remove dots from front of a sym
dropDots:{`$((s in .Q.an)?1b)_s:string x}

/ tree recursion
tree:{$[x~k:key x;x;11h=type k;raze (.z.s ` sv x,) each k;()]}

/ generate list of directories
dirs:{(` sv x,) each key[x] except `q`Q`h`j`o`prof`debug`qml}

/ generate list of profileable functions
lambdas:{x where 100h=(type get@) each x} 

/ instrument all functions
modifyAllFuncs:{modifyFunc each lambdas except[raze tree each `.,dirs`;`..lg`..lgts]}


