OS 버전: Ubuntu 18.04
Python 버전: 3.8.17

코드를 실행하기에 앞서, requirement.txt에 있는 라이브러리를 설치합니다.

[모델 출처]
- unet 기반 모델의 경우 pretrained-backbones-unet 라이브러리를 사용하였습니다.
https://github.com/mberkay0/pretrained-backbones-unet
- upernet 기반 모델의 경우 transformers 라이브러리를 사용하였습니다.
https://huggingface.co/openmmlab/upernet-convnext-large
 
[Training]
- {model_name}_train.py는 트레이닝 코드입니다.
- torch.cuda.set_device(0) 부분을 GPU환경에 맞게 수정합니다.
- model_name = "/home/ubin108/data/unet_efficientnetV2_l_final.pth" 을 가중치 경로에 맞게 수정합니다.

[Inference]
- inference.ipynb 파일에 해당합니다. 
- 각 모델과 모델에 맞는 가중치를 불러와 실행하는 코드입니다.
- 각 코드블럭에서 가중치 경로 및 csv 파일 경로를 설정하고 실행하시면 됩니다. 

[Ensemble]
- ensemble.ipynb 파일에 해당합니다.
- Inference과정으로부터 생성된 csv 파일로부터 최종 앙상블을 진행합니다.
- 각 csv 파일 경로를 설정하고 코드를 실행하시면 됩니다.