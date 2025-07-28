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


python train.py --model-name vim --model-class Vim --ckpt-name Vim --epochs 80
python train.py --model-name yolo5 --model-class Yolo5 --ckpt-name Yolo5 --epochs 80

python train.py --model-name vit1 --model-class Vit1 --ckpt-name Vit1 --epochs 80

python train.py --model-name diffusion --model-class Diffusion --ckpt-name Diffusion --epochs 80
python train.py --model-name diffusion --model-class Diffusion --ckpt-name Diffusion_3cnn --epochs 80


python train.py --model-name DeepUNet_PS8x_16to128 --model-class DeepUNet_PS8x_16to128 --ckpt-name DeepUNet_PS8x_16to128 --epochs 80
python train.py --model-name erspn --model-class ERSPN --ckpt-name ERSPN --epochs 80

python train.py --model-name dff_net --model-class DFF_Net --ckpt-name DFF_Net --epochs 80
python train.py --model-name hasrn --model-class HASRN --ckpt-name HASRN --epochs 80

python train.py --model-name dhunet --model-class DHUnet --ckpt-name DHUnet --epochs 80
