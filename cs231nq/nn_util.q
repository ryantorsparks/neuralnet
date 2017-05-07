/ utils
system"c 30 150"

/ for comments, trim off white spaces,
/ then shift everything to the right two spaces
lg:{-1 $[10h=type x;"\n" sv "  ",/:ltrim each "\n" vs $[x like "#*#";"\n",x,"\n";x];.Q.s x];};

/ util to convert n dimentional q matrix to python copy paste-able
/ (useful for confirming results should match)
bracket:{"[",x,"]"}
toPythonArray:{{"np.array(",x,")"} {$[all 0h=type each x;bracket "," sv .z.s each x;bracket "," sv x]} {$[type x;bracket "," sv string x;.z.s each x]}x}

/ function to set an array in kdb from python message
/ example, call from python:
/ >>> from qpython import qconnection
/ >>> q = qconnection.QConnection(host = 'localhost', port = 5000, username = 'tu', password = 'secr3t', timeout = 3.0)
/ >>> q.open()
/ >>> q('setPythonArray', 'x',x.shape, x.flatten())
/ >>> q('setPythonArray', 'w',w.shape, w.flatten())
/ >>> q('setPythonArray', 'b',b.shape, b.flatten())
/ can also wrap up on python side into a function, e.g.:
/ >>> def setQVar(name,var):
/       q('setPythonArray', name,var.shape, var.flatten())
/       return name
/ >>> setQVar('dx_num',dx_num)
setPythonArray:{[name;dim;x](`$name) set dim#x}

/ (taken from nick psaris' funq github)
shape:{$[0h>t:type x;();n:count x;n,.z.s x 0;1#0]} 

/ reshape function
reshape:{[x;w](count[x],count w)#razeo x}

/ equivalient to flip rehsape[x;w], but about 10-15% faster for typical use case here
flipReshape:{[x;w]razeo[x] @ ((cw*til cx:count x)+/:til cw:count w) }

/ matrix multiply, use qml if possible
dot:@[{system"l qml.q";lg"setting dot as qml.mm";.qml.mm};();{"no qml, dot is mmu";mmu}];

/ relative errors
relError:{[x;y]max/[abs[x-y]%1e-8|sum abs(x;y)]}

/ generate linear space (np.linspace)
/ evenly spaced numbers
/ e.g. linSpace[-0.5;0.5;12]
linSpace:{[start;end;n]start+((end-start)%n-1)*til n}

/ log space function, like linspace but 10 xexp [linSpace]
logSpace:{[start;end;n] 10 xexp linSpace[start;end;n]}

/ save assignment inputs
psave:{[name;param] (`$":assignmentInputs/fullyConnected_",string name) set param}

/ same as np.random.randn, generates random arrays with var and dev = 1.0
/ and avg=0.0
pi:3.1415926535897931
randArray:{(x;y)#sqrt[-2*log n?1.]*cos[2*pi*(n:x*y)?1.]}

/ raze over
razeo:raze/

/ array standard dev
adev:{dev razeo x}

/ sum over 
sumo:sum/

/ null dictionary
nulld:enlist[`]!enlist(::)

/ transform dict `dx`dw`db!(a;b;c) -> `dx3`dw3`db3!(a;b;c) (for input layer 3)
renameKey:{[layer;dict] (`$1_'string[key dict],\:string layer)!value dict};

/ get model params
/ e.g. getModelValue[d;`params]
/      getModelValue[d;`init]
getModelValue:{[d;x]
    if[not `model in key d;'"getModelValue: d is missing `model from key"];
    modelFunc: ` sv d[`model],x;
    if[not count key modelFunc;'"getModelValue: model function ",(-3!modelFunc)," does not exist"];
    modelFunc@d
 };

/ dictionary default get, try and get a value from a dictionary, 
/ if it's not present, then use a default value
dget:{[d;v;default]
    $[v in key d;d v;default]
 };

