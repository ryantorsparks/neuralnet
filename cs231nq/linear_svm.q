\l nn_util.q

svmLossVectorized:{[w;x;y;delta]
	 scores:dot[x;w];
	 correctScores:scores@'y;
	 margins:0|scores-correctScores-delta;
	 loss:(.5*reg*r$r:raze w)+sum/[margins]%count x;
	 xmask:"f"$margins>0;
	 xmask:@'[xmask;y;:;neg sum each xmask];
	 dw:(reg*w)+(dot[flip[x];xmask])%count x;
	 (loss;dw)
  };
