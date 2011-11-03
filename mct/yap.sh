for (( i=0; $i<40; i=$i+1 ))
do
  echo  -*-*-*-*-*-*-*-*-*
done

cd /home/can2/code/mct

rm *.o *.mexa64;  
rm /tmp/kek

matlab -nodesktop -nosplash -r "run_nmf ; exit"&> /tmp/kek
#matlab -nodesktop -nosplash -r "run" &> /tmp/kek

#less /tmp/kek
reset
