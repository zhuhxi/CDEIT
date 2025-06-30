python train.py --model-name ecnet --model-class EcNet --ckpt-name EcNet --epochs 80
python train.py --model-name eitnet --model-class EITNet --ckpt-name EITNet --epochs 80
python train.py --model-name sadb_net --model-class SADB_Net --ckpt-name SADB_Net --epochs 80
python train.py --model-name cnneim --model-class CNN_EIM --ckpt-name CNN_EIM --epochs 80
python train.py --model-name srcnn --model-class SRCNN --ckpt-name SRCNN --epochs 80
python train.py --model-name cunet --model-class CUnet --ckpt-name CUnet --epochs 80
python main.py --data-path ./data --results-dir result_zhx --mode train
python train.py --model-name cdeit --model-class CDEIT --ckpt-name CDEIT --epochs 80
python train.py --model-name cdeit_ecnet --model-class CDEIT_ECNET --ckpt-name CDEIT_ECNET --epochs 80

python train.py --model-name vit --model-class Vit --ckpt-name Vit --epochs 80
python train.py --model-name vivim --model-class Vivim --ckpt-name Vivim --epochs 80


python train.py --model-name vim --model-class Vim --ckpt-name Vivim --epochs 80