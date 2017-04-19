\l nn_util.q
cifarMode:@[value;`cifarMode;{`flattened}]

dataDir: `:CIFAR_data;
dataDirUnflattened: `:CIFAR_data_unflattened;
binDataDir: ` sv dataDir,`$"cifar-10-batches-bin";
cifarDataUrl: "http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";
binData:`data_batch_1.bin`data_batch_2.bin`data_batch_3.bin`data_batch_4.bin`data_batch_5.bin`test_batch.bin
binFile:`$"cifar-10-binary.tar.gz"

/ function to download and load CIFAR 10 data, it's ugly and messy due to laziness/sloppiness, and 
/ also having to keep things memory light for 32 bit kdb, otherwise w-aborts happen
loadCIFARBinaryData:{[]
    lg "not all data found, check if we need to download";
    if[not all binData in key binDataDir;
        lg "not all binary data objects found, checking if the .tar.gz file exists";
        if[not binFile in key dataDir;
            lg "no ",string[binFile]," file found, begin download of cifar 10 data, this may take a while (162 MB)";
            (` sv dataDir,binFile) 1: .Q.hg hsym `$cifarDataUrl;
          ];
        lg "extracting data using ",cmd:"tar -xzvf ",(1_ string ` sv dataDir,binFile)," -C CIFAR_data/";
        system cmd;
      ];
    lg "loading in CIFAR binary data, reshaping into objects of 3073 length";
    allData::raze {0N 3073#read1 ` sv binDataDir,x} each binData;

    lg "get training and validation data";
    yTrain::`int$49000#allData[;0];
    xTrain::{flip 0N 1024#x}each `real$1_'49000#allData;
    yVal::`int$-1000#allData[;0];
    xVal::{flip 0N 1024#x}each `real$1_'-1000#allData;
    
    lg "clearing data, for 32 bit memory issues";
    delete allData from `.;
    .Q.gc[];
    
    lg "subtracting means from xTraining data";
    lg "getting averages for training data";
    avgRes::{`real$avg xTrain[;x]}each til 1024;
    {if[0=x mod 100;lg "subtracting mean for index ",string[x]," of 1024"];xTrain[;x]:xTrain[;x]-\:avgRes x;}each til 1024;
    xTrain::raze each xTrain;
    
    lg "mean subtraction for validation data";
    {if[0=x mod 100;lg "subtracting mean for index ",string[x]," of 1024"];xVal[;x]:xVal[;x]-\:avgRes x;}each til 1024;
    xVal::raze each xVal;
 
    lg "same load and reshape for test data";
    allData::0N 3073#read1 ` sv binDataDir,`test_batch.bin;
    yTest::`int$allData[;0];
    xTest::{flip 0N 1024#x}each `real$1_'allData;
    delete allData from `.;
    .Q.gc[];
    
    lg "mean subtraction for testing data";
    {if[0=x mod 100;lg "subtracting mean for index ",string[x]," of 1024"];xTest[;x]:xTest[;x]-\:avgRes x;}each til 1024;
    xTest::raze each xTest;
    delete avgRes from `.;

    saveTab[dataDir;] each vars;
 };


reshapeTab:{[tabName]
    lg "reshaping ",string tabName;
    {[tabName;x]if[0=x mod 500;show x];
       .[`.;(tabName;x);{32 32#/:x@(3*til 1024)+/:til 3}];
    }[tabName;] each til count value tabName;
 };

loadCIFARBinaryDataUnflattened:{[]
    lg "loading unflattened cifar binary data";
    / first see what's missing    
    missingXVars:`xTrain`xTest`xVal except key dataDirUnflattened;
    missingYVars:`yTrain`yTest`yVal except key dataDirUnflattened;
    if[count missingYVars;
        lg"copying over y objects: ",-3!missingYVars;
        {lg cmd:"cp "," " sv {1_string ` sv x,y}[;x]each dataDir,dataDirUnflattened;system cmd} each missingYVars
      ];
    lg "not all unflattened data found, lets reshape from the flat versions: ",-3!missingXVars;
    loadAllTabs[dataDir];
    reshapeTab each missingXVars;
    saveTab[dataDirUnflattened;]each missingXVars;

 };

/ save and load functions
saveTab:{[dir;x]lg "saving ",string[x]," to ",string[dir]," for future quick loading";(` sv dir,x) set value x}
loadAllTabs:{[dir]{[x;dir]lg "loading ",string x;load ` sv dir,x}[;dir] each vars where not vars in key `.} 

lg "first check if data exists in CIFAR_data folder, if so we don't need to download and process";
loadDir:$[cifarMode~`unflattened;dataDirUnflattened;dataDir];
lg "load dir set as ",string loadDir;

$[all (vars:`xTrain`yTrain`xTest`yTest`xVal`yVal) in key loadDir;
    [ lg "data exists in kdb form on disk, loading directly";
      loadAllTabs[loadDir];
    ];
    [ if[not all vars in key dataDir;loadCIFARBinaryData[]];
      if[cifarMode~`unflattened;loadCIFARBinaryDataUnflattened[]];
      loadAllTabs[loadDir]
    ]
  ];

/ not sure why this is needed, but dot[;w] type errors otherwise
xVal:`real$xVal
xTest:`real$xTest
