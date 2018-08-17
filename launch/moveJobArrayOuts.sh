#!/bin/bash

for f in `ls out.*.*.o`
do

DIR0=`head -n 1 $f`
DIR1=${DIR0::-6}
ERRF=${f::-1}e
#echo $ERRF
mv $f $DIR1
mv $ERRF $DIR1

done
