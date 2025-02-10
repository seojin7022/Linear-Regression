# Linear Regression(선형 회귀)
Linear regression은 $(x, y)$라는 데이터들이 $y \approx w_1x + w_0 $의 선형 관계를 갖는다고 가정하여 최적의 $(w^\*_0, w^\*_1)$ 을 찾는 것이다. (일변수 함수를 예시로 들었다.)

## MSE Loss(Mean Squared Estimation Loss)
MSE Loss는 예측값과 실제값의 차이를 제곱하여 더하고 평균을 낸 것이다.

즉 이것을 가지고 모델이 얼마나 잘 예측하였는지 평가할 수 있고, 이 Loss의 그래디언트를 이용해 W를 업데이트 해 나가면 최적의 $\mathbf{w}$를 구할 수 있다.

$$L = \frac{1}{N}\sum^{N}_{i=1}(\mathbf{w}^T\mathbf{x}^{(i)} - y^{(i)})^2$$

$$\nabla_w L = \frac{2}{N}\sum^{N}_{i=1} (w^Tx^{(i)} - y^{(i)})x^{(i)}$$

## Gradient Descent(경사 하강법)
MSE Loss를 그래디언트 한 것($\nabla_wL$)을 이용해 Loss를 최소화 시키는 방향으로 $w$를 업데이트 하는 것이다.

$$w \coloneqq w - \alpha \nabla_wL$$

여기서 $\alpha$는 learning rate라고 불리는데, 빠른 학습 속도와 제대로 된 학습을 위해서는 이 값을 적절히 조율해야한다.

(너무 크게 설정 할 경우 Loss 값이 무한대로 발산할 수 있다.)

## 코드 설명
numpy 라이브러리를 사용하고, 벡터를 이용하여 기존 for문을 사용하는 것 보다 매우 빠르게 계산하였다.
