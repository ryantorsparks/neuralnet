{load ` sv `:CIFAR_data,x} each `xTrain`xVal`yTrain`yVal
xVal:"f"$xVal / not sure why this is needed, but fails during predict otherwise

/ binary stuff
/
alldata:raze {0N 3073#read1 hsym `$"data_batch_",string[x],".bin"}each 1+til 2
yTrain:`int$49000#alldata[;0]
xTrain:{flip 0N 1024#x}each `int$1_'49000#alldata
