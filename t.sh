#!/bin/sh
count=0
for i in 2 4 6
do 
echo "i is $i"
count=`expr $count + 1`
done
echo "The loop was executed $count times"

case 1 in
   1) echo "First line"
   ;;
   2) echo "SEcond line"
   ;;
   3) echo "3rd line"
   ;;
esac
