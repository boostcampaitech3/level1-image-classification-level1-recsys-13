# level1-image-classification-level1-recsys-13

## Folder Structure

```shell
  SeongBeomLEE
  ├── jupyter
  ├── model.py
  ├── loss.py
  ├── metric.py
  ├── dataAugmentation.py
  ├── dataPreprocessing.py
  ├── dataset.py
  ├── transform.py
  ├── train.py
  ├── requirements.txt
  └── README.md
```

## Use

```shell
pip install -r requirements.txt

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
