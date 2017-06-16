/ loading in cifar data
\l nn_util.q

dataDir: `:CIFAR_data_razed;
binDataDir: ` sv dataDir,`$"cifar-10-batches-bin";
cifarDataUrl: "http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";
binData:`data_batch_1.bin`data_batch_2.bin`data_batch_3.bin`data_batch_4.bin`data_batch_5.bin`test_batch.bin
binFile:`$"cifar-10-binary.tar.gz"

/ function to download and load CIFAR 10 data, it's ugly and messy due to laziness/sloppiness, and 
/ also having to keep things memory light for 32 bit kdb, otherwise w-aborts happen
loadCIFARBinaryData:{[]
    lgts "not all data found, check if we need to download";
    if[not all binData in key binDataDir;
        lgts "not all binary data objects found, checking if the .tar.gz file exists";
        if[not binFile in key dataDir;
            lgts "no ",string[binFile]," file found, begin download of cifar 10 data, this may take a while (162 MB)";
            (` sv dataDir,binFile) 1: .Q.hg hsym `$cifarDataUrl;
          ];
        lgts "extracting data using ",cmd:"tar -xzvf ",(1_ string ` sv dataDir,binFile)," -C ",1_string ` sv dataDir,`;
        system cmd;
      ];
    lgts "loading in CIFAR binary data, reshaping into objects of 3073 length";
    allData::raze {0N 3073#read1 ` sv binDataDir,x} each binData;
    
    lgts "get training and validation data";
    yTrain::`int$49000#allData[;0];
    xTrain::{flip 0N 1024#x}each `real$1_'49000#allData;
    yVal::`int$-1000#allData[;0];
    xVal::{flip 0N 1024#x}each `real$1_'-1000#allData;
    
    lgts "clearing data, for 32 bit memory issues";
    delete allData from `.;
    .Q.gc[];
    
    lgts "subtracting means from xTraining data";
    lgts "getting averages for training data";
    avgRes::{`real$avg xTrain[;x]}each til 1024;
    .Q.gc[];
    {if[0=x mod 100;lgts "subtracting mean for index ",string[x]," of 1024"];xTrain[;x]:xTrain[;x]-\:avgRes x;}each til 1024;
    xTrain::raze each xTrain;
    
    lgts "mean subtraction for validation data";
    {if[0=x mod 100;lgts "subtracting mean for index ",string[x]," of 1024"];xVal[;x]:xVal[;x]-\:avgRes x;}each til 1024;
    xVal::raze each xVal;
 
    lgts "same load and reshape for test data";
    allData::0N 3073#read1 ` sv binDataDir,`test_batch.bin;
    yTest::`int$allData[;0];
    xTest::{flip 0N 1024#x}each `real$1_'allData;
    delete allData from `.;
    .Q.gc[];
    
    lgts "mean subtraction for testing data";
    {if[0=x mod 100;lgts "subtracting mean for index ",string[x]," of 1024"];xTest[;x]:xTest[;x]-\:avgRes x;}each til 1024;
    xTest::raze each xTest;
    delete avgRes from `.;

    reshapeTab each vars where vars like "x*"; 
//    saveTab[dataDir;] each vars;
    saveTabChunks[dataDir;]each vars;
 };


/ reshape, one at a time to save RAM
reshapeTab:{[tabName]
    lgts "reshaping ",string tabName;
    {[tabName;x]if[0=x mod 500;show x];
       .[`.;(tabName;x);{x@razeo (3*til 1024)+/:til 3}];
    }[tabName;] each til count value tabName;
 };

/ save and load functions
saveTab:{[dir;x]lgts "saving ",string[x]," to ",string[dir]," for future quick loading";(` sv dir,x) set value x}
saveTabChunks:{[dir;x]
    if[0<type value x;:saveTab[dir;x]];
    lgts "saving ",string[x]," to ",string[dir]," using chunks for future quick loading";
    n:count value x;
    chunk:100;
    / get all indices for each chunk
    savepath:` sv dir,x;
    inds:til[n div chunk]*chunk;
    {[x;savepath;chunk;ind]lgts"saving ",string[x]," at index ",-3!ind;savepath upsert `float$x@ind+til chunk}[x;savepath;chunk;]each inds;
    lgts "finished saving ",string[x],", clearing for RAM";
    delete x from `.;
    .Q.gc[];
 };

loadAllTabs:{[dir]{[x;dir]lgts "loading ",string x;load ` sv dir,x}[;dir] each vars where not vars in key `.} 

lgts "first check if data exists in CIFAR_data folder, if so we don't need to download and process";
loadDir:dataDir;
lgts "load dir set as ",string loadDir;

$[all (vars:`xTrain`yTrain`xTest`yTest`xVal`yVal) in key loadDir;
    [ lgts "data exists in kdb form on disk, loading directly";
      loadAllTabs[loadDir];
    ];
    [ if[not all vars in key dataDir;loadCIFARBinaryData[]];
      loadAllTabs[loadDir]
    ]
  ];

