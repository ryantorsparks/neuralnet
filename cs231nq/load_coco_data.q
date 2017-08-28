/ loading in cifar data
\l nn_util.q
lgts "loading coco data"
dataDir:`:COCO_data

downloadCocoData:{system"wget -r --ftp-user=rspa9428 --ftp-password=\"Pa\\$\\$W0rd1234\" ftp://ftp.drivehq.com/cs231n/COCO_data.tgz"};
unzipCocoData:{system"tar -xzvf COCO_data.tgz"}
cocoFiles:`train_captions`train_features`train_image_idxs`val_captions`val_features`val_image_idxs`idx_to_word`word_to_idx;

getCocoFiles:{
    if[all cocoFiles in key `:COCO_data;:()];
    if[not `COCO_data.tgz in key `:.;downloadCocoData[]];
    unzipCocoData[];
 };

getCocoFiles[];  
     
(load ` sv dataDir,)each cocoFiles; 

train2014_urls:read0 ` sv dataDir,`train2014_urls.txt
lgts "finished loading coco data"
/

dataDir: `:COCO_data;
binDataDir: ` sv dataDir,`$"cifar-10-batches-bin";
cifarDataUrl: "http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";
cocoDataUrl:"http://cs231n.stanford.edu/coco_captioning.zip";
cocoFile:`coco_captioning.zip
cocoFiles:`coco2014_captions.h5`coco2014_vocab.json`train2014_images.txt`train2014_urls.txt`train2014_vgg16_fc7.h5`train2014_vgg16_fc7_pca.h5`val2014_images.txt`val2014_urls.txt`val2014_vgg16_fc7.h5`val2014_vgg16_fc7_pca.h5
binFile:`$"cifar-10-binary.tar.gz"

/ function to download and load CIFAR 10 data, it's ugly and messy due to laziness/sloppiness, and 
/ also having to keep things memory light for 32 bit kdb, otherwise w-aborts happen
loadCOCOData:{[]
    lgts "not all data found, check if we need to download";
    if[not all cocoFiles in key binDataDir;
        lgts "not all binary data objects found, checking if the .tar.gz file exists";
        if[not cocoFile in key dataDir;
            lgts "no ",string[cocoFile]," file found, begin download of COCO data, this may take a while (987 MB)";
            system"wget ",cocoDataUrl," -P ",1_string ` sv dataDir,`;
          ];
        lgts "extracting data using ",cmd:"unzip ",(1_ string ` sv dataDir,cocoFile)," -C ",1_string ` sv dataDir,`;
        system cmd;
      ];
    lgts "loading in CIFAR binary data, reshaping into objects of 3073 length";
    allData::raze {0N 3073#read1 ` sv binDataDir,x} each binData;
    
    reshapeTab each vars where vars like "x*"; 
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

