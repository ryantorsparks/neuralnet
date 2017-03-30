{load ` sv `:CIFAR_data,x} each `xTrain`xVal`yTrain`yVal
xVal:"f"$xVal / not sure why this is needed, but fails during predict otherwise
