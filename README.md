# REC 가격 예측 모델

## 📌 프로젝트 개요
- 한국 REC 가격의 변동성을 예측하고자 함
- Transformer, LSTM기반 모델을 통해 REC매수 조언

## 🗂️ 사용한 데이터
- REC 가격 데이터 (`data/rec.csv`)
- 전력 수요 데이터 (`data/demand.csv`)
- REC 발급량 (매년 2월부터 이전년도 데이터 사용)
- RPS 비율


## 🛠️ 사용 기술
- TensorFlow, Keras, WSL (모델 훈련 환경)
- 커스텀 Loss, 커스텀 Accuracy 기반 Checkpoint 콜백

## 📈 모델 구조
- Input -> LSTM -> Attention -> Dense -> Output
- Input shape: (32, 16)
- Output: -1, 1 사이값

## 🦾 훈련
- Custom Loss를 사용해 초기 학습 진행 후, MSE로 fine-tuning
 - 처음부터 MSE로 학습하면 예측값이 0에 수렴하는 경향
- Accuracy 기준으로 모델을 저장하는 커스텀 Checkpoint 사용

## 📊 추론 결과
![Image](https://github.com/user-attachments/assets/2298f896-93c6-4c59-8c7e-36b07cee1ab7)
- 초록색 구간: 매수 추천
- 빨간색 구간: 매도 추천

## ⚠️ 한계
- Accuracy 기준이 직관적이지 않음 (예측값과 y값의 오차가 0.2 미만일 경우 정답 처리)
- REC 거래 데이터의 양이 부족함
- 추세보다는 재생에너지 정책의 영향을 크게 받을 수 있음

