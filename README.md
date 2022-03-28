# pre_onboarding_assignment_3_STS

## 1. 데이터 분석 및 True Summary

### 1-1. sports_news_data

- 약 6개월 간 축구 뉴스 기사 데이터
- Title: 기사 제목
- Content : 기사 내용
- Publish_DT : 기사 발행일 ( 사용하지 않음 )

### 1-2. 데이터 EDA

- 총 9077개의 데이터로 이루어져있습니다.
- null 데이터 2개, 본문 중복 데이터 25개
- 본문이 1줄로만 이루어져있는 데이터 3개

- 기사 특징 파악

특징 파악을 위해 제목, 본문, 리드(전문)형태로 데이터를 분리하여 사용했습니다.
    
    
    - 평균 어절 개수 : 제목(9.4개), 본문(197.9개), 전문(32.3개)
    - 평균 명사 포함률(명사 개수/전체품사개수) : 제목(48%), 본문(34%), 전문(39%)
    - 기사 본문 문장 수 시각화 (20-30 문장의 사이가 가장 분포가 많다)
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e5c9b178-824e-486b-8078-86a71de99248/Untitled.png)
    

### 1-3. True Summary 설정 이유

저희 팀의 True Summary는 상황에 맞는 유연한 y값을 가지기 위하여 차등적으로 설정됩니다.

1. **TextRank** : 대부분의 데이터의 True Summary를 단어 빈도 통계(TF-IDF)를 사용하는 알고리즘 TextRank를 사용하여 True Summary로 사용하였습니다.
2. **전문** : 본문의 토큰수가 7개 이하인 경우 TextRank를 사용할 수 없었고 해당되는 본문의 개수는 150여개 였습니다. 그렇기 때문에 150여개 데이터에 대해서는 전문(본문 앞의 3문장)을 사용하여 True Summary로 지정하였습니다. 
3. **제목** : korbertsum을 사용하기 위해서는 3개의 True Summary가 필요하기 때문에 150여개의 본문 중 본문 문장 개수가 3이하인 경우는 모델을 사용하여도 요약의 의미가 없다고 판단하여 기사의 특징 중 요약의 기능을 맡은 첫번째 문장을 출력하도록 설정하였습니다.

- 제목을 True summary로 사용하지 않은 이유 : 평균 어절 개수가 본문 데이터에 비해 5%정도 밖에 되지 않아 제대로 된 요약을 내놓을 수 없다는 판단과 평균 품사 포함률에서도 요약을 해야하는 본문(34%)과 제목(48%)은 상당한 차이를 보임으로 생략되는 품사가 많아 부적합하다고 판단하였습니다.
- 전문을 주된 True summary로 사용하지 않은 이유 : 팀에서 사용하는 모델은 kobertsum, matchsum으로 input에 summary의 index가 포함되어야 합니다. 그러나 전문을 index로 사용할 경우에는 {0,1,2}와 같이 똑같은 index의 값이 입력되게 됩니다. 해당의 경우 딥러닝은 블랙박스로 어떻게 학습이 이루어질지 판단이 불가한 상태여서 단순히 index를 학습할 수도 있다고 판단하여 전체적으로 사용할 수없었습니다.
- 150여개 데이터에 전문을 True summary로 사용한 이유 : 논문에 따르면 요약task를 수행할 때, 기사의 구조는 두괄식이기 때문에 전문을 사용할 경우TextRank 보다 정확도가 높은 성능을 내기도 합니다. 그렇기 때문에 9000여개 데이터에서 150여개 정도의 경우에 불가피하게 전문을 true 값으로 사용할 수 있다고 판단하였습니다. ([http://koreascience.or.kr/article/JAKO202024852036141.pdf](http://koreascience.or.kr/article/JAKO202024852036141.pdf))
- TextRank를 주된 True summary로 사용한 이유 : 아래 그림은 TextRank를 수행하였을 시에 출력되는 sports 데이터의 summary index 분포입니다. 전문을 사용할 경우와 다르게 본문의 중반에서도 많은 분포를 사용하는 것을 확인하였기에 딥러닝 input으로 적합하다고 생각하였습니다.

![TextRank알고리즘 - sports_news_data의 summary index 분포](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/280c6fe7-6941-484b-8c45-34ccc2574c5f/Untitled.png)

TextRank알고리즘 - sports_news_data의 summary index 분포

### 1-4. 데이터 전처리

- html 문법 글자 삭제
- 000기자, 통신원 삭제
- 의미 없는 띄어쓰기, 들여쓰기 와 같은 공백 제거
- 이미지 URL, [스포탈코리아]와 같은 데이터 삭제
- 온점으로 문장을 분리할때 스포츠 기사의 특성같이 평점(8.58점)과 같이 작성되는 경우가 많기에 예외로 처리하여 문장을 분리

## 2. 프로젝트 구조 및 모델 설명

### 2-1. ****KoBertSum****

- 문장을 요약문에 포함시킬지에 대한 여부를 이진분류합니다. 문장들을 대표하는 후보를 선정할 때 두번째, 세번째 후보를 뽑을때는 이전 후보와는 유사하지 않은 문장을 뽑습니다.
- 기존 bert와 다르게 2-3개의 transformer layer를 추가 하여 사용하며 2문장까지만 입력가능했던 것과 달리 여러 문장을 입력하여 각 문장 앞에 생성된 [CLS]토큰의 벡터값을 사용하여 요약문에 적합한지를 binary classification하여 사용합니다.

### 2-2. Matchsum

- MatchSum은 Extractive Summarization을 목적으로 하며, Source Document에서 독립적으로 문장을 추출하고, 문장들 사이의 관계를 모델링합니다.
- Candidate를 추출하고, Matching하는 두 단계로 구성된 모델입니다.
- Rouge Score를 이용하여 Sentence Level Score와 Summary Level Score를 정의한 뒤, 문장 단위로 보았을 때는 점수가 낮은 문장을, 요약문 전체를 보았을 때는 점수가 높은 문장인 Pearl summary를 정의하여 사용합니다. 이러한 Pearl Summary를 탐지하기 위해 Candidate Summary와 Source Document를 Weight를 공유하는 Siames-BERT에 입력으로 사용하여 Cosine Similarity를 가장 크게 할 수 있는 방식으로 학습을 수행합니다.

### 2-3. Metric 설정

- rouge1, Rouge2, Rougel의 평균 값으로 메트릭을 정의했습니다.
- ROUGE는 기본적으로 참조요약과 모델요약의 겹치는 단어 수를 세서 지표화 한것으로, n-gram, longest sentence, skip-gram같은 방법을 사용해서 다방면으로 문장이 유사한지 알아봅니다.
- 단순히 unigram이 겹치는 정도로만 유사성을 보는 것이 아니라 여러가지 방법을 섞어서 사용해서 각각의 모자란부분을 보완한다고 생각해서 모델로 추출한 요약이 얼마나 참조요약과 비슷한지 알기 위해 ROUGE score를 사용했습니다.

### 2-4. 모델 선정 이유 및 모델 설계 구조

1. TextRank를 사용하여 기본적인 True Summary 설정하였습니다.

2. input으로 기존 데이터와 TextRank를 사용하여 얻은 True Summary를 사용합니다. KoBertSum을 사용하여 각 문장의 임베딩값을 사용하여 TextRank보다 문장단위에서 더 좋은 요약문의 형태를 출력할 수 있기 때문에 모델을 선정하였고 추가적인 모델인 Matchsum의 input으로도 활용할 수 있다는 장점이 있습니다.

3. KoBertSum의 output을 다시 Matchsum의 input으로 사용합니다. Matchsum을 사용하여 문장단위 이외에 요약문 전체적으로도 유사도를 활용하여 이전 모델의 output보다 사람이 인식하기에 의미론적으로도 매끄러운 요약문을 출력함으로 전체적인 구도를 잡아 3가지의 모델이 시너지 효과를 낼 수 있다고 판단하여 순차적으로 모델을 적용하는 방식으로 구조를 잡았습니다.

### 2-5. 모델 훈련

- AI_HUB의 문서요약 텍스트/ 신문기사train/valid(총24만개) 데이터를 추가 데이터로 사용했습니다.
- KoBERT pretrained 모델에 transformer encoder를 2개와 sigmoid를 씌워 summary task를 수행할 수 있도록 구현하였습니다.
- train -loss를 줄이며 학습하고 save된 check point들을 하나씩 validation set으로 loss를 체크해보며 최적점을 확인하였으며 loss가 고착된 후 파라미터들을 수정하며 학습을 시도해보았지만 loss는 큰 변화가 없었습니다.

## 3. 하이퍼 파라미터 튜닝

### 3-1. 하이퍼 파라미터 전 base_line

base line : { rouge-1 F : 0.460 , rouge-2 F : 0.308 , rouge-l F : 0.386} = metric {0.896}

base line는 sports_news_data의 7000개 데이터를 사용해서 학습했을 때 출력된 결과입니다.

### 3-2. 하이퍼 파라미터 튜닝

![ai_hub 데이터 optuna 실행 결과](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8dbc33d8-1f59-4688-991d-6bb957adf5c6/Untitled.png)

ai_hub 데이터 optuna 실행 결과

1. Optuna 라이브러리를 사용하여 random search 방법을 사용했습니다.
2. 24만개의 ai_hub를 입력하여 위의 그림과 같이 random 설정된 경우의 내용을 시각화여 loss가 작은 값이 많이 선택하는 하이퍼파라미터의 수치를 확인하여 훈련하는 방법으로 진행하였습니다.
3. 그렇게 적용된 하이퍼파라미터에서 batch_size의 경우 300으로 채택되어 훈련을 진행했으나 24만개의 데이터에 비해 너무 작은 batch size로 빠른 과적합 및 전체 데이터를 효율적으로 활용하지 못한다고 판단하여 batch size를 늘려가며 훈련을 해봤습니다.

Ai_hub 데이터의 test 결과

- batch_size 300 - { rouge-1 F : 0.451 , rouge-2 F : 0.293 , rouge-l F : 0.372} = metric {0.868}
- batch_size 1000- { rouge-1 F : 0.484 , rouge-2 F : 0.336 , rouge-l F : 0.408} = metric {0.956}

### 3-3. 하이퍼 파라미터 튜닝 결과

- Optuna 의 best parameter
    - batch_size=1000
    - dropout=0.1
    - lr=0.00025
    - warmup=0.1

튜닝 결과 :  { rouge-1 F : 0.492 , rouge-2 F : 0.345 , rouge-l F : 0.416} = metric {0.975}

## 4. 개인 담당 역할

- sports_news_data의 EDA 분석 및 True summary 설정을 하였습니다.
- Matchsum 영어 버전을 한국어 버전으로 변경하여 실행 할 수 있도록 분석하고 수정하는 전체적인 과정을 수행하였습니다.
- kobertsum의 하이퍼 파라미터 튜닝 과정을 진행하며 best 파라미터를 찾았습니다.

## 5. 최종 결과 분석 및 후기

base line : { rouge-1 F : 0.460 , rouge-2 F : 0.308 , rouge-l F : 0.386} = metric {0.896}

최종 결과 : { rouge-1 F : 0.492 , rouge-2 F : 0.345 , rouge-l F : 0.416} = metric {0.975}

최종 결과는 Ai_hub 데이터(240,000개)를 통해 추가 학습을 하고 optuna를 통해 best parameter를 적용한 값입니다. 학습한 데이터의 차이가 있지만 DACON에서 진행한 한국어 문서 추출요약 AI 경진대회에 leaderboard와 큰 차이를 보이지 않습니다. 결과값의 경우에는 True summary로 설정하였던 TextRank와 다른 출력을 내는 것을 확인할 수 있었고 중요한 내용을 내포하는 것을 확인하였습니다.

물론 처음 설계하였던 Matchsum까지 적용하여 전체적인 내용이 유사한 요약문을 출력할 수 있었으면 더할 나위 없었겠지만 시간이 부족하여 Train을 넘지 못하여 아쉬운 점이 큽니다. 그러나 영어 모델을 한국어 모델로 변경하기 위해 전체적으로 어떤 부분을 신경써야하는지, 무엇을 먼저 확인해야할지, 영어와 한국어에서 코드 차이가 어떻게 생기는지 에 대한 인사이트를 많이 얻었기 때문에 충분히 가치있는 시간이었습니다.
