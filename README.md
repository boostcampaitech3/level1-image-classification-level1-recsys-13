## 01. 프로젝트 개요

### (1) 프로젝트 주제 및 개요


<p align="center"><img src="https://user-images.githubusercontent.com/52911772/156746616-5235f462-5cde-4a03-8dfc-429f846602cc.png" width="400" height="300"/></p>


- 본 프로젝트는 위 그림 처럼 한 사람의 이미지가 주어졌을 때 해당 사람의 마스크 착용 여부(올바른 착용, 잘못된 착용, 미착용), 나이(청년층, 중년층, 노년층), 성별(남, 여) **총 18개의 label을 예측하는 최적의 모델**을 만드는 것이다.


<p align="center"><img src="https://user-images.githubusercontent.com/52911772/156746783-81d03677-d784-46ab-9a25-44ad138986b2.png" width="700" height="300"/></p>


- 본 데이터는 **매우 불균형한 형태의 데이터**라고 할 수 있으며, **나이를 제대로 맞추는 것이 이번 대회의 핵심**이라고 할 수 있었다. 실제로 사람의 눈으로도 해당 사람의 나이를 구분하기 어려운 경우가 많았다.

### (2) 활용 장비 및 재료(개발 환경 등)

### **협업 툴**

<p align="center"><img src="https://user-images.githubusercontent.com/52911772/156746872-d09207d0-7f3a-4ac2-b3af-9bde5ce41eef.png" width="400" height="300"/></p>

## 02. 프로젝트 팀 구성 및 역할

- 팀 구성 및 역할
    - 신민철 : EDA, 모델 성능 평가,  Multi Model 실험, Ensemble, 실험일지 작성
    - 유승태 : 모델 구현, 모델 성능 평가, Data Augmentation, Multi Model 실험, Ensemble
    - 이성범 : 모델 구현. Data Augmentation, 파이프라인 구축, Multi Model 실험, Ensemble
    - 최종현 : EDA, 모델 성능 평가, Data Augmentation, Multi Model 실험, 실험일지 작성
    - 한광석 : EDA, 모델 성능 평가, Data Augmentation, 실험일지 작성

## 03. 프로젝트 수행 절차 및 방법

<p align="center"><img src="https://user-images.githubusercontent.com/52911772/156747022-7766dc1d-2206-47ba-a56b-f7ce6f26f18b.png" width="800" height="200"/></p>



## 04. 프로젝트 수행 결과

## (1) Data Processing

- **이상치 처리**
    - 잘못 기재된 Gander, Mask Label에 대한 이상치 처리


- **Class 불균형 문제에 대한 실험**
    - sklearn의 StratifiedKFold를 활용하여 Train과 Val 데이터의 클래스가 균일 하고, **서로 다른 사람이 존재하도록 K-fold Set을 설정**
    - StratifiedSampler을 사용하여 **Batch 단위 데이터의 클래스가 균일**하도록 설정
    - 클래스의 분포에 대한 영향력을 줄이기 위해서 **OverSampling**, DownSampling 등을 실험


- **데이터 변형 및 증강 기법에 대한 실험**
    - 얼굴을 조금 더 잘 인식할 수 있도록 이미지에 얼굴 부분만을 남기고 나머지 부분은 자르는 **FaceCrop** 을 실험
    - 옷의 색깔, 피부색 등 색상에 대한 영향력을 제거하기 의해서 이미지에 색상을 모두 회색으로 바꾸는 **GrayScale** 을 실험
    - 주름 등 나이에 대한 영향력을 주는 요인을 조금 더 잘 파악할 수 있도록 이미지의 해상도를 높이는 **CLAHE(Contrast Limited Adaptive Histogram Equalization)** 을 실험
    - 데이터를 증가하기 위해, 3명의 사람에 대한 이미지의 평균 값을 이용하여 새로운 이미지를 생성하는 **MixUp** 기법을 실험
    - ColorJitter, GaussianBlur, RandomRotation, RandomHorizontalFlip 등 다양한 이미지 변형 기법을 **오버샘플링을 위하여 확률적으로 적용**
    
## (2) Modeling


- **사용 CNN 기반 모델**
    - **원본 이미지 사이즈를 그대로 활용**할 수 있다는 장점
    - 깊은 모델이 곧 높은 성능을 보장하지는 않음 (과적합되는 경향을 보여줌)
    
    | 모델 패키지 | 모델 이름 | 비고 |
    | --- | --- | --- |
    |  torchvision | Resnet18  | baseline |
    |  | EfficientNet B0 |  |
    |  | EfficientNet B4 | B5부터 Batch Size 줄여줘야 함 |
    | timm | RegnetX_002 | RegnetX 계열 사용 채택 |
    |  | RegnetY_002 |  |


- **사용 Transformer 기반 모델**
    - 원본 이미지를 모델에 맞게 resize 해줘야 함 → 성능 하락
    
    | 모델 패키지 | 모델 이름 |
    | --- | --- |
    | timm | Vision Transformer |
    |  | Swin Transformer |


- **Mask / Gender / Age 분류에 대한 실험**

모델 예측 오류의 대부분이 Age에서 발생한다는 것을 확인
- Mask / Gender / Age 각 모델 만든 후 앙상블한 모델
  - 각 특성에 대해서 따로 예측하면 성능이 향상될 것이라 판단


- Age 만 regression
    - 세부적인 age를 잘 예측하기 위해 활용


- 전체 Label 분류 후에 Age 재분류
    - Age 모델의 성능이 낮아서 Age를 잘 분류하기 위해 활용


- 성별에 따른 age 모델 분리
    - 남자 여자의 노화 속도가 다르다는 점 반영하여 각 gender별 age 분류 모델


- Mask → 예측 값 → Gender → 예측 값 → Age → 최종 label 생성 (step by step Model)
    - 각 특성을 단계적으로 예측하면 성능이 향상될 것이라 판단


- 여러 단일 모델 결과의 앙상블(hard voting)
    - 다양한 결과를 종합해 봤을 때 정답에 더 가까워 질 수 있을 것이라 판단
    

- 여러 단일 모델의 softmax 값을 lgbm을 앙상블(soft voting style...)
    - csv 앙상블과 마찬가지로 결과를 종합했을 때 정답에 가까워 질 것이라 판단

    
- **Multi Sample Dropout 적용 실험**
    - 모델 마지막 부분에 Dropout layer를 추가하여 weight 동조 현상을 방지
    

- **Pretrained 적용 실험**

    | 실험 방식 | 비고 |
    | --- | --- |
    | Not-Pretrained | pretrained 모델에 비해 더 많은 epoch 학습 필요  |
    | Backbone Freeze |  |
    | Pretrained | 사용 채택 |


## (3) Training

- **Loss Function 실험**
    - **클래스의 분포에 모델의 학습 속도를 조정**하기 위해 **Weighted Cross Entropy Loss, Focal Loss** 등을 실험
    - **Metric에 적합한 모델**을 만들기 위해서 전체 label을 골고루 맞힐 수 있는 **Label Smoothing, Focal Loss, F1 Loss** 등을 실험


- **Optimizer 실험**
    - Adam을 사용시 train loss는 계속 감소하지만, val loss는 계속하여 증가한다는 문제점을 발견(pre-trained 모델 사용시 초기 Momentum 정보를 잊어버림)
    - **AdamW, RAdam 등으로 변경하여 모델의 학습 속도 조절**하고, 과적합을 줄여 train loss와 val loss 사이의 차이를 줄임


- **모델 학습 속도 조정 실험**
    - Learning Rate와 Batch Size 크게 설정할 시 loss 값이 빠르게 0에 수렴하고 Overshooting이 발생하는 문제점 발견
    - **Learning Rate와 Batch Size를 작게 설정**함으로써 모델의 학습 속도를 늦추고 조금 더 일반화된 모델을 얻음


- **기타 모델 성능 향상을 위한 실험**
    - **Test Time Augmentation(TTA)를 활용**하여 서로 다른 데이터 변형에 따른 결과를 Ensemble 함으로써 모델의 표현력을 높임
    - **Stochastic Weight Averaging(SWA)를 활용**하여 모델의 Weight를 동적이게 하여 모델의 표현력을 높임
    - **Pseudo Labeling 기법을 활용**하여 모델의 결정 경계를 조금 더 확실히 하고, 학습 데이터의 양을 늘릴 수 있도록 함


## 04. 자체 평가 의견


<p align="center"><img src="https://user-images.githubusercontent.com/52911772/156746171-8092f900-0bef-4bd8-b536-4913944bcbe8.png" width="700" height="300"/></p>



- Baseline을 빠르게 구축을 하여 다양한 실험을 진행하면서 검증 데이터 셋에 대하여 성능이 향상되었다. 하지만, 모델의 성능을 올리는 것에만 집중하여 결국 리더보드의 점수는 계속 침체기에 빠지게 되었고, 프로젝트에서 좋은 성적을 얻지 못했다.


- 실험 일지 작성이 미흡했다. 실험을 진행할 때마다 해당 실험 기록을 진행 해야 했다. 실험 일지 작성을 안 한 결과 앞으로 진행할 실험의 방향을 잃어버렸다. Wandb를 활용해서 실험 내역을 정리하면서 실험을 진행했으면 좋았을 것 같다. 실험 일지 작성 미흡이 역할 분담이 잘 안된 문제로까지 이어진 것 같다.


- 테스트 데이터에 대해 너무 맹신한 게 문제였다. 테스트 데이터에 대해 의심을 해 볼 필요가 있었다.
