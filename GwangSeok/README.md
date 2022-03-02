# GwangSeok

* `2022-02-21`
  * 강의: overview, EDA, Dataset
  * 실습: overview, EDA
* `2022-02-22`
  * 강의: Data Generation, Model
  * 실습: 강의 정리, EDA
* `2022-02-23`
  * 강의: Training & Inference,  More...
  * 실습: 이상치 데이터 변환
* `2022-02-24`
  * 실습: regnet-y-400mf, regnet-y-800mf, data augmentation
      * regnet-y-400mf의 성능이 더 좋다.
      * albumentations를 사용할 때, import cvs를 해주는데 docker error가 발생할 수 있다.
* `2022-02-25`
    * 실습: wandb 설정, timm에 있는 모델 실험 최적의 모델 찾기
        * regnety가 제일 적당한 것 같다.
* `2022-02-28`
    * 실습: mixup으로 데이터 증가, 앙상블 방법(stacking: 18label 으로 구분 후 0~6 다시 구분)
        * 성능이 그다지 높지 않다.
* `2022-03-02`
    * 코드를 공부하면서 직접 preprocessing과 dataset, dataloader를 만들었다.
    * 내일은 모델을 만들고 실습을 해야겠다.