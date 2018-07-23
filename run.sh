# !/bin/bash

export PATH=$PATH:/home/slb/anaconda2/envs/py36_tf_keras/bin
#use envs’s python and its lib
export PYTHONPATH=/home/slb/anaconda2/envs/py36_tf_keras:$PYTHONPATH
export LD_LIBRARY_PATH=/home/slb/anaconda2/envs/py36_tf_keras/lib:$LD_LIBRARY_PATH

python --version

cd /home/slb/Desktop/Project/src

if [$# -lt 0];then
echo 'please input right parameter.'
exit 0
fi

if [$# -eq 1];then
echo 'please input right parameter.'
exit 0

elif  [$# -eq 2];then
echo 'Begin to predict:'
echo 'test data path :':$1
echo 'output csv result path :' $2
python /home/slb/Desktop/Project/src/main.py $1 $2

else
echo 'Begin to predict:'
echo 'test data path :':$1
echo 'output csv result path :' $2
echo 'select gpu ID :' $3
python /home/slb/Desktop/Project/src/main.py  $1 $2 --gpus $3 
echo '####################next is prediction#############################'
fi

#$1 and $2 is position parameters,must give a value!
##example:sh /home/slb/Desktop/Project/run.sh /home/slb/Desktop/data/Attributes/Round2b/ /home/slb/Desktop/Project/
##example:sh /home/slb/Desktop/Project/run.sh /home/slb/Desktop/data/Attributes/Round2b/ /home/slb/Desktop/Project/  2
##example:nohup sh /home/slb/Desktop/Project/run.sh /home/slb/Desktop/data/Attributes/Round2b/ /home/slb/Desktop/Project/  3 > /home/slb/Desktop/log.txt 2>&1

