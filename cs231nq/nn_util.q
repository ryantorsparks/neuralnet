/ utils
/ for comments, trim off white spaces,
/ then shift everything to the right two spaces
lg:{-1 $[10h=type x;"\n" sv "  ",/:ltrim each "\n" vs x;.Q.s x];};


/ util to convert q matrix to python copy paste-able
/ (useful for confirming results should match)
bracket:{"[",x,"]"}
toPythonArray:{"np.array(",(bracket "," sv {bracket "," sv string x} each x),")"}

/ (taken from nick psaris' funq github)
shape:{$[0h>t:type x;();n:count x;n,.z.s x 0;1#0]} 

/ matrix multiply, use qml if possible
dot:@[{system"l qml.q";lg"setting dot as qml.mm";.qml.mm};();{"no qml, dot is mmu";mmu}];
