/ debugging funcs, rarely needed (hopefully)
/ will print out time, .Q.w[`used]*1e-9, i.e. memory used in GB
/ @param x - string to log to
.debug.lg:{lg (string[.z.t]," : ",string 1e-9*.Q.w[]`used)," GB - ",x}  

/ takes a given function name, modifies it to print its own name first (along with mem used),
/ used for debugging w-aborts
/ @param x - function name as a symbol
/ e.g:
/   q) f:{100+x*y+z}
/   q) .debug.modifyFunc[`f]
/   q) f[3;4;5]
/     00:20:43.386 : 0.000336688 GB - calling function `f
/   127


.debug.modifyFunc:{[funcName]
    if[not 100h=type value funcName;:()];
    (`$".old.",string funcName) set value funcName;
    funcName set (')[{[f;n;p] .debug.lg "calling function ",-3!n;f . p}[value funcName;funcName];enlist]
 };

.debug.modifyAllFuncs:{.debug.modifyFunc each system["f"]except `lg}


