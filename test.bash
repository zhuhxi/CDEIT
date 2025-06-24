python test.py --model-name ecnet --model-class EcNet --ckpt-name EcNet
python plot.py --ckpt_dir /home/zhx/word/work/CDEIT/checkpoints/EcNet --num_samples 5

python test.py --model-name eitnet --model-class EITNet --ckpt-name EITNet
python plot.py --ckpt_dir /home/zhx/word/work/CDEIT/checkpoints/EITNet --num_samples 5

python test.py --model-name sadb_net --model-class SADB_Net --ckpt-name SADB_Net 
python plot.py --ckpt_dir /home/zhx/word/work/CDEIT/checkpoints/SADB_Net --num_samples 5

python test.py --model-name cnneim --model-class CNN_EIM --ckpt-name CNN_EIM 
python plot.py --ckpt_dir /home/zhx/word/work/CDEIT/checkpoints/CNN_EIM --num_samples 5

python test.py --model-name srcnn --model-class SRCNN --ckpt-name SRCNN
python plot.py --ckpt_dir /home/zhx/word/work/CDEIT/checkpoints/SRCNN --num_samples 5

python test.py --model-name cunet --model-class CUnet --ckpt-name CUnet
python plot.py --ckpt_dir /home/zhx/word/work/CDEIT/checkpoints/CUnet --num_samples 5

python test.py --model-name cdeit --model-class CDEIT --ckpt-name CDEIT
python plot.py --ckpt_dir /home/zhx/word/work/CDEIT/checkpoints/CDEIT --num_samples 5

python main.py --mode test --data simulated --results-dir results_zhx