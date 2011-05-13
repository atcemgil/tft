for (( i=0; $i<40; i=$i+1 ))
do
  echo  -*-*-*-*-*-*-*-*-*
done

cd /home/can2/code/mex_cudatensor

rm cudatensor.o cudatensor.mexa64;  
rm /tmp/kek

#VAR=$1
VAR=2

matlab -nodesktop -nosplash -r "testcuda($VAR)"&> /tmp/kek
less /tmp/kek
#cat /tmp/kek
