for i in 0 1 2 3 4
do
    accelerate launch --mixed_precision=fp16 train_script.py --fold $i
done