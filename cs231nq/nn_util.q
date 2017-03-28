/ utils
dot:@[{system"l qml.q";show"setting dot as qml.mm";.qml.mm};();{"no qml, dot is mmu";mmu}];

/ util to convert q matrix to python copy pastable
bracket:{"[",x,"]"}
toPythonArray:{"np.array(",(bracket "," sv {bracket "," sv string x} each x),")"}

shape:{$[0h>t:type x;();n:count x;n,.z.s x 0;1#0]} 
