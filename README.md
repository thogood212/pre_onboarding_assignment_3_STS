# pre_onboarding_assignment_3_STS

## 1. 데이터 설명 ( KLUE/STS Dataset )

### 1-1. KLUE/STS Dataset

- **문장 유사도 비교(STS)** : Airbnb, PARICY, PARAKQC를 이용하여 다양한 의미적 문맥을 포함하고 있습니다. 의도 의문과 PARAKQC의 주제 정보는 의미론적으로 관련된 문장 쌍을 생성할 때 유용합니다.
- **Policy News (POLICY) :** POLICY는 대한민국 부처, 국가청, 국가위원회가 배포하는 다양한 문서 데이터 세트입니다. 2020년 말까지 발표된 기사들을 포함합니다.
- **ParAKQC(PARAKQC) :** PARAKQC는 스마트홈 기기를 겨냥한 10,000개의 발화 데이터셋으로, 10개의 유사한 의문에 대해 1,000개의 의도로 구성되어 있습니다. 스마트 홈 스피커와 교류할 때 가능한 다양한 주제를 다룹니다.
- **Airbnb Reviews (AIRBNB) :** AIRBNB는 AIRBNB 홈페이지에서 공개적으로 접속할 수 있는 리뷰 데이터 세트 입니다. AIRBNB에의해 수집되고 사전 처리된 기존의 다국어 AIRBNB 리뷰에서 시작합니다.

### 1-2. KLUE/STS Dataset 구성

![[출처] KLUE 공식 벤치마크 사이트](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4f80ff8f-37f6-4148-a1e8-745e5171a997/Untitled.png)

[출처] KLUE 공식 벤치마크 사이트

- 총 13,224개의 데이터로 이루어져있습니다.
- STS를 두 입력 문장의 의미적 유사성을 0(의미 중복 없음)에서 5(의미 동등성)까지 라벨링 되어있습니다. 모델 성능은 Pearson의 상관 계수로 측정됩니다.
- 실수를 임계값 점수가 3.0인 두 개의 클래스로 이진화하고(의역 여부) F1 점수를 사용하여 모델을 평가합니다.

## 2. 모델 설명

### 2-1. Pre_trained 모델 설명 : KLUE/RoBERTa

![[출처] KLUE 논문 : [https://arxiv.org/pdf/2105.09680.pdf](https://arxiv.org/pdf/2105.09680.pdf)](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7dbfccf1-d5e7-4c97-9c9a-39b850ab1494/Untitled.png)

[출처] KLUE 논문 : [https://arxiv.org/pdf/2105.09680.pdf](https://arxiv.org/pdf/2105.09680.pdf)

- 학습한 코퍼스 크기 : 위의 그림의 데이터셋에서 코퍼스를 가져와 사용하였습니다. Volume 열의 값이 small : 1k 미만 corpus, midium : 1k ~ 50k corpus, Large : 50k 이상 corpus크기를 가지고 있습니다.
- 선택 이유 : 기존 BERT보다 더많은 데이터를 사용, Dynamic Masking를 수행하여 문장 유사성 부분에서 SOTA을 도달한 모델이기 때문에 RoBERTa 모델의 한국어 버전인 KLUE/RoBERTa를 선택하였습니다.

### 2-2. Pre_trained 모델 사용 모듈 : **Hugging Face**

- 다양한 트랜스포머 모델(*transformer.models*)과 학습 스크립트(*transformer.Trainer*)를 제공하는 모듈인 허깅페이스를 활용하여 STS 문장유사도를 해결하였습니다.
- sentence-transformers 보다 하이퍼 파라미터 튜닝 및 학습에 더 강점이 있다고 판단하여 허깅페이스 모듈을 선택하였습니다.

### 2-3. Pre_trained 모델 학습 방법

- Tokenizer(KLUE/RoBERTa)를 사용하여 512의 길이로 데이터셋 안의 sentence1,2를 전처리 합니다.
- 허깅페이스 Trainer를 사용하며 훈련하며 사용하는 metric은 compute_metrics 함수를 정의하여 f1 score가 최대가 되도록 학습하였습니다.
- Defalt로 batch_size=32, epoch =10, seed=42로 고정하여 진행하였습니다.

## 3. 하이퍼 파라미터 튜닝

### 3-1. 하이퍼 파라미터 전 base_line

![validation 실행 시 F1 : 0.94 , Pearsonr : 0.95](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f677cf4a-3466-4ede-801b-36b7543bb72b/Untitled.png)

validation 실행 시 F1 : 0.94 , Pearsonr : 0.95

![Test 실행 시 F1 : 0.82 , Pearsonr : 0.84](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6a1aab13-13b0-4fb5-9960-c72140979269/Untitled.png)

Test 실행 시 F1 : 0.82 , Pearsonr : 0.84

### 3-2. 하이퍼 파라미터 튜닝 순서

1. **Seed** : random seed 마다 score의 차이가 있기 때문에 그 중 가장 좋은 결과를 얻기 위해 seed를 탐색하여 가장 높은 점수가 나온 seed = 42로 고정하여 사용합니다.
2. **학습 방법 A** : 
- Learning Rate scheduler를 linear, step scheduler로 각각 사용하여 차이를 확인합니다.
- 성능이 좋았던 scheduler를 사용하여 batch_size를 16,32,64로 변경하여 각각 연산합니다.
- weight decay를 추가하여 성능 변화를 확인합니다.

1. **학습 방법 B** : 
- grid_search를 사용하여 Learning Rate 최적값을 찾습니다.
- batch_size를 16,32,64로 변경하여 최적값을 찾습니다.
- weight decay를 추가하여 성능 변화를 확인합니다.

1. Optuna를 통해 랜덤으로 찾아낸 best 파라미터와 A,B 방법을 통해 찾은 best 파라미터의 성능을 비교하여 최적값을 파악합니다.

### 3-3. 하이퍼 파라미터 튜닝 결과

- Optuna1 의 best parameter
    - learning rate: 1.4654261946051012e-05
    - batch_size: 8
    - weight_decay: 0.061673419363888114
    - warmup_steps: 0.01
- A 모델 의 best parameter
    - lenarning rate: iner schedule
    - batch_size = 32,
    - weight decay: 0.01
- B 모델 의 best parameter
    - learning rate: 5e-5
    - batch_size: 64
    - weight_decay: 0.01

![3개의 모델의 결과값 확인](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/834d7da5-cbc0-4e05-a45f-835153ec92b1/Untitled.png)

3개의 모델의 결과값 확인

비교 결과 : 각기 다른 방법으로 하이퍼 파라미터를 튜닝해보았고 그 중 직접 값을 입력하여 결과를 살펴본 A,B모델보다 random search의 Optuna 라이브러리를 사용하는 것이 F1, pearsonr score를 골고루 좋은 성능을 내었기 때문에 Optuna로 찾은 파라미터로 최종 결정 하였습니다.

## 4. 개인 담당 역할

- huggingface 모듈을 찾아 훈련까지의 전과정을 찾고 선행 진행하였습니다.
- Optuna를 사용하여 초기의 하이퍼파라미터 최적값을 찾아보았습니다.
- REST API를 Flask_app을 통하여 구현하였습니다.

![[그림1]](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d31358c6-37f9-43c2-b0a8-027e4f9618c5/Untitled.png)

[그림1]

그림1의 초기창에서

유사한 문장인 문장을 입력하였을 때 그림2 결과창이 출력됩니다.

> 문장1 :최근 국민들의 여행심리 위축 등으로 동남아 등 다른 노선까지 영향을 받는 상황이다.
문장2 :동남아시아와 같은 다른 노선은 최근 사람들의 여행 감정의 하락에 영향을 받았습니다.
> 

유사하지 않은 문장을 입력하였을 때 그림3 결과창이 출력됩니다.

> 문장1 : 학생들의 균형 있는 영어능력을 향상시킬 수 있는 학교 수업을 유도하기 위해 2018학년도 수능부터 도입된 영어 영역 절대평가는 올해도 유지한다.
문장2 :영어 영역의 경우 학생들이 한글 해석본을 암기하는 문제를 해소하기 위해 2016학년도부터 적용했던 EBS 연계 방식을 올해도 유지한다.
> 

## 5. 최종 결과 분석

![[그림2] 유사한 문장을 입력한 결과창](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b6c81a04-5b6e-416e-92b3-7d33d3d24044/Untitled.png)

[그림2] 유사한 문장을 입력한 결과창

![[그림3] 유사하지 않은 문장을 입력한 결과창](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/df1d2756-234f-4453-b8ae-60a31aa3f611/Untitled.png)

[그림3] 유사하지 않은 문장을 입력한 결과창

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/df702998-9198-41f6-995b-573d6ccf6548/Untitled.png)

위의 그림은 저희가 설정한 base line과 하이퍼 파라미터 튜닝 후의 모델의 성능을 비교한 그래프입니다.

F1 score의 경우 0.82에서 0.87까지 성능이 올랐으며, Pearsonr score의 경우 0.84에서 0.88까지 성능을 끌어올렸습니다.

F1 score에 비해 Pearsonr score의 성능 증가폭이 적을 뿐만 아니라 KLUE github에 공시되어있는 base line보다 F1 score는 높으나 Pearsonr score가 낮은 결과를 보여주는 한계점이 있습니다.

batch_size를 설정함에 있어서 과적합 혹은 undertrain 현상이 일어날 것이라 판단하여 직접 하이퍼 파라미터 튜닝을 할 때, 16이하의 batch_size를 사용하지 않았지만 해당 task에서는 비교적 적은 데이터로 진행되었다는 점과 epoch가 적게 돌아가도 최적성능을 내는 점을 통하여 설정했던 batch_size보다 작은 수로 적용했어야 한다는 아쉬움이 남습니다.

하지만 결론적으로 목표로 하였던 F1 score가 공시된 base line보다 높게 결과를 출력하였기 때문에 추가 보완을 거치면 좋은 성능을 기대할 수 있겠습니다.
