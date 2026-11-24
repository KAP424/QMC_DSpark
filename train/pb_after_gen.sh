#!/bin/bash
# Wait for Target generation to finish, verify the data is complete, then start the 5-epoch training.
W=/home/dev/KAP/data/pb_dspark
D=/home/dev/KAP/train
PY=/home/dev/venv/bin/python
while pgrep -f '[p]b_gen_worker.py' > /dev/null; do sleep 30; done
echo "$(date) generation workers exited" >> $W/pipeline.log
if $PY -I $D/pb_check_gen.py >> $W/pipeline.log 2>&1; then
  echo "$(date) data complete, starting training" >> $W/pipeline.log
  CUDA_VISIBLE_DEVICES=0 $PY -I $D/pb_train.py --epochs 5 --temp 1.0 >> $W/train_stdout.txt 2>&1
  echo "$(date) training process exited with status $?" >> $W/pipeline.log
else
  echo "$(date) data incomplete, training NOT started" >> $W/pipeline.log
fi
