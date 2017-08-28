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
