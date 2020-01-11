패션 MNIST를 이용한 옷분류 모델
=============
>### 목표

&nbsp;&nbsp;- Keras의 패션 MNIST 데이터셋을 훈련시켜 옷분류 예측모델을 만든다

&nbsp;&nbsp;
>### 패션 MNIST 데이터셋

- 10개의 범주(category)와 70,000개의 흑백 이미지로 구성된 데이터셋

- 이미지는 해상도(28x28 픽셀)가 낮고 다음처럼 개별 옷 품목을 나타낸다

![데이터셋](https://user-images.githubusercontent.com/52990642/72201197-ddd8d280-3494-11ea-9b2a-f0f26e779a2a.png)

- 레이블은 0~9까지의 정수배열이고, 옷의 클래스를 의미한다

<img width="400" alt="레이블" src="https://user-images.githubusercontent.com/52990642/72201221-1bd5f680-3495-11ea-91f4-5324689d137f.PNG">

&nbsp;&nbsp;
>### 데이터 전처리

- 훈련 세트에 있는 첫 번째 이미지를 보면 픽셀 값의 범위가 0~255 사이라는 것을 알 수 있다
- 신경망 모델에 주입하기 전에 이 값의 범위를 0~1 사이로 조정한다
- 훈련 세트와 테스트 세트를 동일한 방식으로 전처리 한다
   
![전처리](https://user-images.githubusercontent.com/52990642/72201276-8ab34f80-3495-11ea-85a6-53c208e0ddf8.png)


&nbsp;&nbsp;
>### 모델 훈련

- 모델이 이미지와 레이블을 매핑하는 방법을 배운다
- 모델이 훈련되면서 손실과 정확도 지표가 출력된다 
- 이 모델은 5번의 훈련을 통해 0.8757(87%) 정도의 정확도를 달성했다

![모델 훈련](https://user-images.githubusercontent.com/52990642/72201363-8cc9de00-3496-11ea-86fe-1825b28b87d2.PNG)

&nbsp;&nbsp;
>### 최종 예측 결과

![예측결과](https://user-images.githubusercontent.com/52990642/72201380-b8e55f00-3496-11ea-91f9-b04bc0c68e92.png)
