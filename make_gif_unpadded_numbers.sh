!#/bin/bash

imgname=$1
imgtype=$2
gifname=$3

convert -delay 8 \
-density 100 \
-quality 1 \
$(for i in $(seq 0 `ls *${imgname}*.${imgtype} | wc -l`); do echo ${imgname}_${i}.${imgtype}; done) -loop 0 ${gifname}.gif
