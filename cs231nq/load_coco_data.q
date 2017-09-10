/ loading in cifar data
\l nn_util.q
lgts "loading coco data"
dataDir:`:COCO_data

// download from a free ftp hosted site - the original data was in some funny python format
// and I wasn't sure how to convert that directly into kdb, so used qpython to send to q
// then saved as q files and uploaded to the drivehq site, the password and user below are needed
// (it's a free site for now, and have none of my details in there besides my email address, so 
// don't mind sharing this link/user/password with the kx community)
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

train_urls:read0 ` sv dataDir,`train2014_urls.txt
val_urls:read0 ` sv dataDir,`val2014_urls.txt
lgts "finished loading coco data"
