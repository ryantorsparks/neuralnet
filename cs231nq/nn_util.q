/ utils
system"c 30 150"

/ try and load andrey zohlos' qml library
@[{system"l qml.q";lg"loading qml lib";};();{-1 "no qml lib found in $QHOME";}];

/ for comments, trim off white spaces,
/ then shift everything to the right two spaces
lg:{-1 $[10h=type x;"\n" sv "  ",/:ltrim each "\n" vs $[x like "#*#";"\n",x,"\n";x];.Q.s x];};

/ log time and space
/ print (time timeSinceLastPrint MEM(GB) msg)
lgts:{prevNow:@[get;`now;0Nt];now::.z.t;lg " " sv (string .z.t;string now-prevNow;.Q.f[2;1e-9*.Q.w[]`heap]," GB";x)};

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

/ matrix overload of where, from nick psaris' funq
mwhere:{$[type x;where x;(,') over til[count x]{enlist[count[first y]#x],y:$[type y;enlist y;y]}'.z.s each x]}

/ (taken from nick psaris' funq github)
shape:{$[0h>t:type x;();n:count x;n,.z.s x 0;1#0]} 

/ reshape function
reshape:{[x;w](count[x],count w)#razeo x}
reshapeM:{[m1;m2Shape] m2Shape#razeo m1}

/ equivalient to flip rehsape[x;w], but about 10-15% faster for typical use case here
flipReshape:{[x;w]razeo[x] @ ((cw*til cx:count x)+/:til cw:count w) }

/ generate dot index versions of a matrix shape
/ e.g. a 2x3 matrix -> (0 0;0 1;0 2;...;2 0;2 1;2 2)
matrixInds:(cross/)til each

/ very slow dumb way of an ndimensional transpose,
/ probably not a necessary function, if really need to do this, should implement
/ in c
flipn:{[m;flipInds] newshape#razeo[m] @.[matrixInds shapem;(::;flipInds)]?matrixInds[newshape:(shapem:shape m)@flipInds]}

/ matrix multiply, use qml if possible
//dot:@[{system"l qml.q";lg"setting dot as qml.mm";.qml.mm};();{lg "no qml, dot is mmu";mmu}];
dot:@[value;`.qml.mm;{lg"no qml, dot set as mmu";mmu}];

/ hyperbolic tan func, tanh
tanhq:{(1-e)%1+e:exp neg 2*x};
tanh:@[value;"{shape[x]#.qml.tanh razeo x}";{lg"no qml, tanh defined in q";tanhq}];

/ relative errors
relError:{[x;y]max/[abs[x-y]%1e-8|sum abs(x;y)]}

/ generate linear space (np.linspace)
/ evenly spaced numbers
/ e.g. linSpace[-0.5;0.5;12]
linSpace:{[start;end;n]start+((end-start)%n-1)*til n}

/ log space function, like linspace but 10 xexp [linSpace]
logSpace:{[start;end;n] 10 xexp linSpace[start;end;n]}

/ e.g. ppath[`fullyConnected;`affineForward]
/ -> `:assignmentInputs/fullyConnected_affineForward
ppath:{[module;paramName] ` sv `:assignmentInputs,`$"_" sv string module,paramName}

/ save assignment inputs
/ e.g psave[`fullyConnected;`affineForward;affineForwardRes]
/ -> saves affineForwardRes to `:assignmentInputs/fullyConnected_affineForward
psave:{[module;paramName;paramValue] ppath[module;paramName] set paramValue}

/ converse of psave, read in a kdb file from the assignment inputs dir
pget:{[module;paramName] get ppath[module;paramName]}

/ same as np.random.randn, generates random arrays with var and dev = 1.0
/ and avg=0.0
pi:3.1415926535897931
randArray:{(x;y)#sqrt[-2*log n?1.]*cos[2*pi*(n:x*y)?1.]}

/ (r)andom (a)rray n-(d)imensional
rad:{[dims](dims)#sqrt[-2*log n?1.]*cos[2*pi*(n:prd dims)?1.]}

/ raze over
razeo:raze/

/ array standard dev
adev:{dev razeo x}

/ sum over 
sumo:sum/

/ max over
maxo:max/

/ max across axes
/ e.g.
/ m is dim 3 4 5 6 7 8
/ maxAxes[m;3 4] is equivalent to m.max(axis=3).max(axis=4) in python
maxAxes:{[m;axes] {[x;ind].[x;ind#(::);max]}/[m;axes]} 

/ creating newAxes, similar to numpy newaxis
/ e.g. a=np.random.randn(5,4,4,4,4,6)
/      resa=a[:, :, :, np.newaxis, :, np.newaxis]
/ is equivalent to 
/      a: rad 5 4 4 4 4 6
/      resa: newAxis[a;3 5]
newAxes:{[m;newAxesInds] {[x;ind] .[x;ind#(::);$[ind=count shape x;(enlist each@);enlist]]}/[m;asc newAxesInds]}
newAxes:{[m;newAxesInds] {[x;ind] sx:shape x;f:$[ind=count sx;(enlist each);enlist];i:$[ind=count sx;ind-1;ind];.[x;i#(::);f]}/[m;asc newAxesInds]}

/ expand a dimension of a tensor
expandDim:{[m;ind;newShape].[m;ind#(::);newShape#]}

/ similar to np.broadcast_arrays, e.g. takes matrixes of shape
/ x: 2 3 1 3; y: 2 1 4 3; and expands x in the 3rd dimension to be
/ 2 3 4 3, and the y in the 2nd dimension to be 2 3 4 3
broadcastArrays:{[x;y]
    xShape:shape x;
    yShape:shape y;
    xCnt:count xShape;
    yCnt:count yShape;
    $[xCnt>yCnt;
        [y:(newshape:(xCnt-yCnt)#xShape)#enlist y;yShape:newshape,yShape];
      yCnt>xCnt;
        [x:(newshape:(yCnt-xCnt)#yShape)#enlist x;xShape:newshape,xShape];
        ];

    xIs1:xShape=1;
    yIs1:yShape=1;

    xChanges:(wx;yShape wx:where xIs1 and not yIs1);
    yChanges:(wy;xShape wy:where yIs1 and not xIs1);
    
    xNew:expandDim/[x;xChanges 0;xChanges 1];
    yNew:expandDim/[y;yChanges 0;yChanges 1];
    (xNew;yNew) 
  };

/ extension on broadcastArrays, for when you have 2 matrixes, one that is already expanded, 
/ another that needs to be expanded to the first ones shape, and an operation you want to do
/ on them (e.g. * or %)
/ e.g. m:50 3 32 32#100000?10.;m2:1 3 1 1#100?10.;
/      broadcastArraysFunc[m;m2;*]
broadcastArraysFunc:{[m;mToExpand;func] func[m;last broadcastArrays[m;mToExpand]]}
/ shortcuts
/ (b)roadcast (a)rray (m)ultiply/(d)ivide/(s)ubtract/(a)dd
bam:broadcastArraysFunc[;;*]
bad:broadcastArraysFunc[;;%]
bas:broadcastArraysFunc[;;-]
baa:broadcastArraysFunc[;;+]

/ sum Axes function, equivalent to np.sum(m,axis=(inds),keepdims=True)
/ e.g. m=rad 2 3 4 5 6 7
/ sumAxesKeepDims[m;3 5] is equivlanet to np.sum(m,axis=(3,5),keepdims=True)
/lg "attempting to load sumAxesKeepDims6d c function, must be a sumAxesKeepDims6d.so object in $QHOME"
/@[{`sumAxesKeepDims6d set `sumAxesKeepDims6d 2:(`sumAxesKeepDims6d;4)};();{lg"WARNING: failed to load sumAxesKeepDims6d.so, will revert to all q version"}];
/sumAxesKeepDimsC:{[m;axes] {[m;axis] sumAxesKeepDims6d[m;@[mShape;axis;:;1]#0f;mShape:shape m;axis]}/[m;asc axes]} 
sumAxesKeepDimsQ:{[m;axes] {[x;ind].[x;ind#(::);(enlist sum@)]}/[m;asc axes]}
sumAxesKeepDims:$[not ()~key `sumAxesKeepDims6d;sumAxesKeepDimsC;sumAxesKeepDimsQ];


/ collapse axes version of sumAxes
/ e.g. m=rad 2 3 4 5 6 7
/ sumAxes[m;3 5] is equivlanet to np.sum(m,axis=(3,5))
sumAxes:{[m;axes]{[x;ind].[x;ind#(::);sum]}/[m;desc axes]} 

/ null dictionary
nulld:enlist[`]!enlist(::)

/ append an int i to a sym s
symi:{[s;i]`$string[s],string i}  

/ append number n to each sym in list of syms
appendNToSyms:{[syms;n] `$string[syms],\:string n}

/ append layer number to each item in key d
renameKey:{[layer;dict] appendNToSyms[key dict;layer]!value dict}

/ transform dict `dx`dw`db!(a;b;c) -> `dx3`dw3`db3!(a;b;c) (for input layer 3)
removeDFromDictKey:{[dict] (`$1_'string key dict)!value dict}
renameGradKey:{[layer;dict]renameKey[layer;removeDFromDictKey dict]}




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

/ zeropad in n dimensions
/ e.g. zeroPad[2 3 4 5#1f;2]
zeroPad:{[x;pad]
    shapex:shape x;
    padf:{y,(til x),y}[;pad#0N];
    c:2<count shapex;
    newShape:(c#shapex),(2*pad)+c _ shape x;
    cntList:$[c;enlist til shapex 0;()],padf each c _ shape x;
    inds:{raze y+/:x*sum not null y}/[cntList];
    newShape#0^razeo[x]@inds
 };

/ dot indexing funtion, doing  (deep matrix) ./:inds is slow, so this
/ just creates the list of inds we need for doing razeo[deep matrix] @list of inds
dotIndexesAsList:{[m;dotInds] sum flip[dotInds]*reverse prds 1,-1_reverse shape m}

/ faster version of doing deepMatrix ./: dotInds
matrixDotInds:{[m;dotInds] razeo[m]@dotIndexesAsList[m;dotInds]}

// stride stuff
/ strides, the number of bytes to step when going through matrix
/ hard coded to only work for longs and floats (8 byte)
/ example:
/   m:{x#prd[x]?1000.}50 3 38 38
/   newshape: 3 7 7 50 32 32
/   strides: 1444 38 1 4332 38 1
/   asStrided[m;newshape;strides]
asStrided:{[m;newshape;strides] newshape#razeo[m]@{raze x+/:raze y}/[reverse[strides]*'til each reverse newshape]}

/ load in col2im c funtion
lg "attempting to load col2im6dInner function, must be a col2im6dInner.so object in $QHOME";
@[{`col2im6dInner set `col2im6dInner 2:(`col2im6dInner;5)};();{lg"WARNING: failed to load col2im6dInner c function"}];

/ load in maskBroadcast6dAxes35  c funtion
lg "attempting to load maskBroadcast6dAxes35 function, must be a maskBroadcast6dAxes35.so object in $QHOME";
@[{`maskBroadcast6dAxes35 set `maskBroadcast6dAxes35 2:(`maskBroadcast6dAxes35;4)};();{lg"WARNING: failed to load maskBroadcast6dAxes35 c function"}];

/ load in sumAxes35KeepDims6dBroadcast c funtion
lg "attempting to load sumAxes35KeepDims6dBroadcast function, must be a sumAxes35KeepDims6dBroadcast.so object in $QHOME";
@[{`sumAxes35KeepDims6dBroadcast set `sumAxes35KeepDims6dBroadcast 2:(`sumAxes35KeepDims6dBroadcast;2)};();{lg"WARNING: failed to load sumAxes35KeepDims6dBroadcast c function"}];
