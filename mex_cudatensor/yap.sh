for (( i=0; $i<40; i=$i+1 ))
do
  echo  -*-*-*-*-*-*-*-*-*
done
rm cudatensor.o cudatensor.mexa64;  
matlab -nodesktop -nosplash -r "test"
