# level1-image-classification-level1-recsys-13

## Members
- 신민철 : EDA, 기본 모델들의 성능 평가, Ensemble
- 유승태 : EDA, Data Augmentation, 기본 모델들의 성능 평가, Multi Model 실험, Ensemble
- 이성범 : EDA, Data Augmentation, 기본 모델들의 성능 평가 및 모델 구현, Framework 구축, Multi Model 실험, Ensemble
- 최종현 : EDA, 기본 모델들의 성능 평가, Ensemble
- 한광석 : EDA, Data Augmentation, 기본 모델들의 성능 평가, Ensemble

## Folder Structure

```shell
  level1-image-classification-level1-recsys-13
  ├── model.py
  ├── loss.py
  ├── metric.py
  ├── dataAugmentation.py
  ├── dataPreprocessing.py
  ├── dataset.py
  ├── transform.py
  ├── train.py
  └── requirements.txt
```

## Use

```shell
python train.py
```

## Config Setting
```shell
config = {
        'facecrop_data' : True,
        'num_classes': 18,
        'num_workers': 4,
        'epochs': 20,
        'batch_size': 64,
        'lr': 9e-05,
        'image_size': [380, 380],
        'image_normal_mean': [0.5, 0.5, 0.5],
        'image_normal_std': [0.2, 0.2, 0.2],
        'timm_model_name': 'regnetx_002',
        'loss': 'cel',
        'oof': 5,
    }
```
