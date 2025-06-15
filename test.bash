python test.py --model-name ecnet --model-class EcNet --ckpt-name EcNet
python plot.py --ckpt_dir /home/zhx/word/work/CDEIT/checkpoints/EcNet --num_samples 5

python test.py --model-name eitnet --model-class EITNet --ckpt-name EITNet
python plot.py --ckpt_dir /home/zhx/word/work/CDEIT/checkpoints/EITNet --num_samples 5

python test.py --model-name sadb_net --model-class SADB_Net --ckpt-name SADB_Net 
python plot.py --ckpt_dir /home/zhx/word/work/CDEIT/checkpoints/SADB_Net --num_samples 5

python test.py --model-name cnneim --model-class CNN_EIM --ckpt-name CNN_EIM 
python plot.py --ckpt_dir /home/zhx/word/work/CDEIT/checkpoints/CNN_EIM --num_samples 5