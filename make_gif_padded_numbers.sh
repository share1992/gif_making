#!/bin/bash

imgname=$1
imgtype=$2
gifname=$3

convert -delay 8 \
-density 100 \
-quality 1 \
${imgname}*.${imgtype} -loop 0 ${gifname}.gif
