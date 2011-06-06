for (( i=0; $i<40; i=$i+1 ))
do
  echo  -*-*-*-*-*-*-*-*-*
done

cd /home/can2/code/mex_cudatensor2

rm cudatensor.o cudatensor.mexa64;  
rm /tmp/kek

matlab -nodesktop -nosplash -r "run"&> /tmp/kek
less /tmp/kek
#cat /tmp/kek
