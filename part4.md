# 출처
> 밑바닥부터 시작하는 딥러닝 3 : https://www.hanbit.co.kr/store/books/look.php?p_code=B6627606922

> 소스코드 : https://github.com/WegraLee/deep-learning-from-scratch-3


# [DeZero] 4. 신경망 만들기


- 목차
    - [1. 텐서 함수 구현](#1-텐서-함수-구현)
    - [2. 신경망 구현(회귀)](#2-신경망-구현회귀)
    - [3. 신경망 구현(다중 클래스 분류)](#3-신경망-구현다중-클래스-분류)
    - [4. 새로운 데이터셋 학습](#4-새로운-데이터셋-학습)


## 1. 텐서 함수 구현
### 원소별 연산을 수행하는 함수
* 텐서 사용 시의 순전파
  * 넘파이의 브로드캐스트(broadcast) : 피연산자의 형상이 다르면 자동으로 데이터를 복사하여 같은 형상의 텐서로 변환해주는 기능
* 텐서 사용 시의 역전파
  * 야코비 행렬(Jacobian matrix) : $\bold{x}, \bold{y}$가 벡터일 때, $\bold{y} = F(\bold{x})$의 미분
    * $
      \frac{\partial \bold{y}}{\partial \bold{x}} = 
      \begin{pmatrix} 
          \frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} & ... & \frac{\partial y_1}{\partial x_n} \\
          \frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} & ... & \frac{\partial y_2}{\partial x_n} \\
          ... & ... & ... & ... \\
          \frac{\partial y_n}{\partial x_1} & \frac{\partial y_n}{\partial x_2} & ... & \frac{\partial y_n}{\partial x_n}
      \end{pmatrix}
      $
  * $1 \times n$ 야코비 행렬(행 벡터) : $y$가 스칼라이고, $\bold{x}$가 벡터일 때, $y = F(\bold{x})$의 미분
    * $\frac{\partial y}{\partial \bold{x}} = (\frac{\partial y}{\partial x_1}, \frac{\partial y}{\partial x_2}, ..., \frac{\partial y}{\partial x_n})$
    * 머신러닝 문제에서는 텐서를 입력받아 스칼라를 출력하는 함수(손실 함수, loss function)를 설정하는 것이 일반적이다.
  * $\bold{a} = A(\bold{x}), \bold{b} = B(\bold{a}), y = C(\bold{b})$ ($\bold{x}, \bold{a}, \bold{b}$는 벡터이고, $y$만 스칼라인 경우)
    * $\frac{\partial y}{\partial \bold{x}} = \frac{\partial y}{\partial \bold{b}} \frac{\partial \bold{b}}{\partial \bold{a}} \frac{\partial \bold{a}}{\partial \bold{x}}$
    * 자동 미분의 forward 모드 : $\frac{\partial y}{\partial \bold{x}} = (\frac{\partial y}{\partial \bold{b}} (\frac{\partial \bold{b}}{\partial \bold{a}} \frac{\partial \bold{a}}{\partial \bold{x}}))$, 
      $
      \frac{\partial \bold{b}}{\partial \bold{x}} = 
      \begin{pmatrix}
          \frac{\partial b_1}{\partial x_1} & \frac{\partial b_1}{\partial x_2} & ... & \frac{\partial b_1}{\partial x_n} \\
          \frac{\partial b_2}{\partial x_1} & \frac{\partial b_2}{\partial x_2} & ... & \frac{\partial b_2}{\partial x_n} \\
          ... & ... & ... & ... \\
          \frac{\partial b_n}{\partial x_1} & \frac{\partial b_n}{\partial x_2} & ... & \frac{\partial b_n}{\partial x_n}
      \end{pmatrix}
      $
    * 자동 미분의 reverse 모드 : $\frac{\partial y}{\partial \bold{x}} = ((\frac{\partial y}{\partial \bold{b}} \frac{\partial\bold{b}}{\partial \bold{a}}) \frac{\partial \bold{a}}{\partial \bold{x}})$, $\frac{\partial y}{\partial \bold{a}} = \begin{pmatrix} \frac{\partial y}{\partial a_1} & \frac{\partial y}{\partial a_2} & ... & \frac{\partial y}{\partial a_n} \end{pmatrix}$
    * 행렬과 행렬의 곱보다 벡터와 행렬의 곱 쪽의 계산량이 더 적다. 따라서 reverse모드의 계산 효율이 더 좋다.

### 원소별로 계산하지 않는 함수
※ 주의 : Function의 backward 내에서 모두 Variable 인스턴스를 사용하므로 구현할 때 **DeZero 함수**를 사용해야 한다.
* 형상 변환 함수
  * `reshape` : 텐서의 형상을 바꾸는 함수 - [[Function으로 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/functions.py#L93) | [Variable 클래스에 추가](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/core.py#L138)]
    * 순전파 : `numpy의 reshape` 함수 사용
    * 역전파 : 기울기의 형상이 입력의 형상과 같아지도록 변환
  * `transpose` : 행렬을 전치해주는 함수 - [[Function으로 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/functions.py#L112) | [Variable 클래스에 추가](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/core.py#L143)]
    * 순전파 : `numpy의 transpose` 함수 사용
    * 역전파 : 출력 쪽에서 전해지는 기울기의 형상을 **순전파 때와 반대 형태**로 변경

* 합계 함수
  * `sum` : 원소가 2개 이상인 벡터의 합 - [[Function으로 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/functions.py#L186) | [Variable 클래스에 추가](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/core.py#L155)]
    * 순전파 : `numpy의 sum` 함수 사용
    * 역전파 : 기울기를 입력 변수의 형상과 같아지도록 복사한다. (아래 `broadcast_to` 함수 사용)
      * axis, keepdim 지원으로 인해 기울기의 형상을 변환하는 경우가 생기기 때문에 이에 대응하는 함수 : [[utils에 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/utils.py#L129)]

* 브로드캐스트 함수
  * `broadcast_to` : x의 원소를 복제하여 shape인수로 지정한 형상이 되도록 한다. - [[Function으로 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/functions.py#L227)]
    * 순전파 : `numpy의 broadcast_to` 함수 사용
    * 역전파 : '원소 복사'가 일어날 경우, 역전파 때는 기울기의 '합'을 구한다. (아래 `sum_to` 함수 사용)
  * `sum_to` : x의 원소의 합을 구해 shape 형상으로 만들어주는 함수 - [[Function으로 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/functions.py#L207)]
    * 순전파 : numpy에 sum_to 함수 없음 - [[utils에 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/utils.py#L108) | [(참고)chainer의 코드](https://github.com/chainer/chainer/blob/master/chainer/utils/array.py#L51)]
    * 역전파 : 입력 x와 형상이 같아지도록 기울기의 원소 복제한다. (위의 `broadcast_to` 함수 사용 (서로 상호의존적))
  * 브로드캐스트 대응
    * 문제 : numpy의 broadcast_to가 보이지 않는 곳에서 이루어지면 역전파가 일어나지 않는다.
    * 해결 : Add, Mul, Sub, Div 등 사칙연산 클래스에서, 순전파 때 broadcast가 일어났다고 판단되면 역전파 때 DeZero의 sum_to를 사용하여 보정한다. - [[core에 반영](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/core.py#L220)]

* 행렬의 곱
  * 개념
    * 벡터의 내적 : 두 벡터 $\bold{a} = (a_1, ..., a_n), \bold{b} = (b_1, ..., b_n)$의 내적 $\bold{ab} = a_1 b_1 + a_2 b_2 + ... + a_n b_n$
    * 행렬의 곱 : 왼쪽 행렬의 '가로 방향 벡터'와 오른쪽 행렬의 '세로 방향 벡터' 사이의 내적을 계산하여 새로운 행렬의 원소가 된다.
      * $\bold{y} = \bold{xW}$, ($\bold{x}$ : $1 \times D$, $\bold{W}$ : $D \times H$, $\bold{y}$ : $1 \times H$)일 때
        * $y_j = x_1 W_{1j} + x_2 W_{2j} + ... + x_i W_{ij} + ... + x_H W_{Hj}$
        * $\frac{\partial y_j}{\partial x_i} = W_{ij}$ 이므로, $\frac{\partial L}{\partial x_i} = {\sum}_j \frac{\partial L}{\partial y_j} \frac{\partial y_j}{\partial x_i} = {\sum}_j \frac{\partial L}{\partial y_j} W_{ij}$, 즉, $\frac{\partial L}{\partial x_i}$은 '벡터 $\frac{\partial L}{\partial \bold{y}}$' 와 '$\bold{W}$의 $i$행 벡터'의 내적으로 구해진다.
        * $\frac{\partial L}{\partial \bold{x}} = \frac{\partial L}{\partial \bold{y}} \bold{W}^T$ ($\frac{\partial L}{\partial \bold{x}}$ : $1 \times D$, $\frac{\partial L}{\partial \bold{y}}$ : $1 \times H$, $\bold{W}^T$ : $H \times D$)
      * $\bold{y} = \bold{xW}$, ($\bold{x}$ : $N \times D$, $\bold{W}$ : $D \times H$, $\bold{y}$ : $N \times H$)일 때
        * $\frac{\partial L}{\partial \bold{x}} = \frac{\partial L}{\partial \bold{y}} \bold{W}^T$
        * $\frac{\partial L}{\partial \bold{W}} = \bold{x}^T \frac{\partial L}{\partial \bold{y}}$
  * `matmul` : 행렬 곱 계산 - [[Function으로 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/functions.py#L257)]
    * 순전파 : `numpy의 dot` 함수 사용
      * `np.dot(x, W)` 대신 `x.dot(W)`로 구현하여 ndarray 인스턴스에도 대응할 수 있도록 한다. 
    * 역전파 : 역전파 계산에서의 행렬의 곱은 `matmul` 함수 사용

* 슬라이스 조작 함수
  * `GetItem`, `get_item` : Variable의 다차원 배열 중에서 일부를 슬라이스(slice)하여 뽑는다. - [[Function으로 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/functions.py#L133) | [Variable 클래스에 추가](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/core.py#L343)]
    * 순전파 : numpy의 ndarray의 slice 기능 그대로 사용
    * 역전파 : 아래의 `GetItemGrad` 함수 이용
  * `GetItemGrad` - [[Function으로 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/functions.py#L147)]
    * 순전파 : `numpy의 add.at` 함수 이용
    * 역전파 : 위의 `get_item` 함수 이용


## 2. 신경망 구현(회귀)
### 선형 회귀
* **회귀**(regression) : x로부터 실숫값 y를 예측하는 것
* **선형 회귀**(linear regression) : 회귀 모델 중 예측값이 선형(직선)을 이루는 것
  * 목표 : $y = Wx + b$, 손실 함수의 출력을 최소화하는 $W$와 $b$ 찾기
  * 평균 제곱 오차(mean squared error) : 선형 회귀는 손실 함수로 평균 제곱 오차를 이용할 수 있다.
    * $L = \frac{1}{N} {\sum}_{i=1}^{N} (f(x_i) - y_i)^2$
    * [[기본 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/functions.py#L418) | [Function으로 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/functions.py#L425)]
* [선형 회귀 구현 예제](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/steps/step42.py)
  * 경사하강법으로 매개변수를 갱신할 때, 계산 그래프를 만들지 않도록 W.data, W.grad.data값을 이용한다.
  * DeZero 내부에서 Variable 인스턴스로 변환하므로 ndarray 인스턴스도 처리할 수 있다. 앞으로도 x가 ndarray 인스턴스인 상태로 들어갈 수 있음

### 신경망
* **선형 변환**(linear transformation) (혹은 아핀 변환(affine transformation))
  * y = F.matmul(x, W) + b
    * W : 가중치(weight), b : 편향(bias)
    * [[기본 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/functions.py#L292) | [Function으로 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/functions.py#L273)]
      * 불필요한 중간 계산 결과의 ndarray 인스턴스는 즉시 삭제하는 것이 바람직하다.
      * [(참고)chainer의 Aggressive Buffer Release](https://docs.google.com/document/d/1CxNS2xg2bLT9LoUe6rPSMuIuqt8Gbkz156LTPJPT3BE/) : 불필요한 ndarray 인스턴스 삭제를 자동화하는 방법
  * 완전연결계층(fully connected layer)에 해당한다.
* **비선형 변환**(nonlinear transformation)
  * 활성화 함수(activation function)
    * ex) ReLU 함수, 시그모이드 함수
  * 시그모이드 함수(sigmoid function)
    * $y = \frac{1}{1 + e^{-x}}$
    * [[기본 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/functions.py#L305) | [Function으로 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/functions.py#L311)]
* [2층 신경망 구현 예제](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/steps/step43.py)
  * 입력층(input layer), 은닉층(hidden layer or middle layer), 출력층(output layer)
  * 가중치의 초깃값은 무작위로 설정하는 것이 좋다.

### 모델
* **Parameter 클래스**
  * 매개변수 : 경사하강법 등의 최적화 기법에 의해 갱신되는 변수 (가중치, 편향)
  * Variable 클래스와 똑같은 기능을 가지지만 매개변수임을 구별할 수 있도록 함
  * [[Parameter 클래스 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/core.py#L167)]
    ```python
    class Parameter(Variable):
        pass
    ```
* **Layer 클래스**
  * Function 클래스와 달리, 매개변수를 유지하고 매개변수를 사용하여 변수를 변환하는 클래스
  * [[Layer 클래스 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/layers.py#L13)]
    ```python
    class Layer:
        def __init__(self):
            self._params = set() # 매개변수 집합(중복ID 저장 방지)

        def __setattr__(self, name, value): # 인스턴스 변수 설정 함수
            if isinstance(value, (Parameter, Layer)):
                self._params.add(name)
            super().__setattr__(name, value)

        def __call__(self, *inputs):
            outputs = self.forward(*inputs)
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            self.inputs = [weakref.ref(x) for x in inputs] # 약한 참조
            self.outputs = [weakref.ref(y) for y in outputs] # 약한 참조
            return outputs if len(outputs) > 1 else outputs[0]

        def forward(self, inputs):
            raise NotImplementedError()

        def params(self):
            for name in self._params:
                obj = self.__dict__[name]

                if isinstance(obj, Layer):
                    yield from obj.params() # Layer 속의 Layer에서 매개변수를 재귀적으로 꺼냄
                else:
                    yield obj # 처리를 '일시 중지(suspend)'하고 값을 반환

        def cleargrads(self):
            for param in self.params(): # `params`메서드를 호출 시 처리를 '재개(resume)'
                param.cleargrad()
    ```
    * `yield`를 사용한 함수를 제너레이터(generator)라고 한다.
    * `yield from`을 통해 또 다른 제너레이터를 만들 수 있다.
  * **Linear 클래스**
    * 계층으로서의 Linear 클래스이며, Layer 클래스를 상속하여 구현한다.
    * [[Linear 클래스 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/layers.py#L89)]
      * `__init__` : Layer 클래스의 `__init__`함수 실행 후, 인수에 따라 가중치와 변수 인스턴스 변수 설정
      * `_init_W` : 가중치 초기화 함수(Xavier initialization)
      * `forward` : 데이터가 흘러오는 시점에 가중치를 초기화할 수 있다.
* **Model 클래스**
  * Layer 클래스를 이용하여 신경망에서 사용하는 매개변수를 한꺼번에 관리할 수 있다.
  * 따라서 Layer 클래스를 상속하여 모델 전체를 하나의 클래스로 정의할 수 있다.
  * [[Model 클래스 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/models.py#L12)]
    ```python
    class Model(Layer):
        def plot(self, *inputs, to_file='model.png'): # 시각화 메서드만 추가
            y = self.forward(*inputs)
            return utils.plot_dot_graph(y, verbose=True, to_file=to_file)
    ```
  * **MLP 클래스**
    * 다층 퍼셉트론(Multi-Layer Perceptron) : 완전연결계층 신경망의 별칭으로 흔히 쓰인다.
    * 범용적인 완전연결계층 신경망을 위한 모델 (TwoLayerNet의 자연스러운 확장)
    * [[MLP 클래스 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/models.py#L32)]
      * `self.l1 = ...` 대신 `setattr`함수를 사용하여 인스턴스 변수 설정

### 최적화
* **Optimizer 클래스**
  * 매개변수 갱신 작업을 모듈화하고 쉽게 다른 모듈로 대체할 수 있는 구조
  * 전처리 수행 함수를 추가할 수 있도록 하면, 가중치 감소(Weight Decay)나 기울기 클리핑(Gradient Clipping) 같은 기법을 이용할 수 있다.
  * [[Optimizer 클래스 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/optimizers.py#L8)]
    ```python
    class Optimizer:
        def __init__(self):
            self.target = None # 매개변수를 갖는 클래스(Model 또는 Layer)
            self.hooks = [] # 전처리를 수행하는 함수들

        def setup(self, target): # target 설정
            self.target = target
            return self

        def update(self): # 모든 매개변수 갱신
            params = [p for p in self.target.params() if p.grad is not None] # grad가 None인 매개변수는 갱신을 건너뛴다.

            # 필요 시 전처리 진행
            for f in self.hooks:
                f(params)

            # 매개변수 갱신
            for param in params:
                self.update_one(param)

        def update_one(self, param): # 구체적인 매개변수 갱신
            raise NotImplementedError()

        def add_hook(self, f): # 원하는 전처리 함수 추가
            self.hooks.append(f)
    ```
  * **SGD 클래스**
    * 확률적 경사하강법(Stochastic Gradient Descent) : 대상 데이터 중에서 무작위로(확률적으로) 선별한 데이터에 대해 경사하강법을 수행한다.
    * Optimizer 클래스를 상속하고 update_one 메서드에서 매개변수 갱신 코드를 구현한다.
    * [[SGD 클래스 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/optimizers.py#L80)]
      ```python
      class SGD(Optimizer):
          def __init__(self, lr=0.01):
              super().__init__()
              self.lr = lr

          def update_one(self, param):
              param.data -= self.lr * param.grad.data
      ```
    * [SGD 클래스 사용 예제](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/steps/step46.py)
  * 이 외
    * [[Momentum 클래스 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/optimizers.py#L89)]
    * [[AdaGrad 클래스 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/optimizers.py#L108)]
    * [[AdaDelta 클래스 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/optimizers.py#L131)]
    * [[Adam 클래스 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/optimizers.py#L160)]


## 3. 신경망 구현(다중 클래스 분류)
### 다중 클래스 분류
* 다중 클래스 분류(multi-class classification) : 분류 대상이 여러 가지 클래스 중 어디에 속하는지 추정
* 소프트맥스 함수(softmax function) : 원소 각각을 확률로 해설할 수 있게 된다.
  * $p_k = \frac{e^{y_k}}{{\sum}_{i=1}^{n} e^{y_i}}$ ($0 \leq p_i \leq 1$, ${\sum}_{i=1}^{n} p_i = 1$)
  * [[기본 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/functions.py#L345) | [Function으로 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/functions.py#L352)]
* 교차 엔트로피 오차(cross entropy error)
  * 원핫 벡터(one-hot vector) : 정답 데이터의 각 원소가 정답에 해당하는 클래스면 1, 아니면 0으로 표현
  * $L = - \underset{k}{\sum} t_k \log p_k$
    * $t_k$ : 원핫벡터로 표현된 정답 데이터의 $k$차원째 값
  * $L = - \log \bold{p}[t]$
    * $t$ : 정답 클래스의 번호
  * [[기본 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/functions.py#L443) | [Function으로 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/functions.py#L454)]
    * $\log (0)$을 방지하기 위해 `clip` 함수 이용
    * [clip 함수](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/functions.py#L644) : Variable x의 원소가 x_min 이하면 x_min으로, x_max 이상이면 x_max로 변환
* [다중 클래스 분류 예제](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/steps/step48.py)
  * [스파이럴(spiral) 데이터셋](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/datasets.py#L43) : 나선형 혹은 소용돌이 모양


## 4. 새로운 데이터셋 학습
### 데이터셋 구조
* **Dataset 클래스**
  * 대규모 데이터셋을 처리할 때, ndarray 인스턴스 하나로 처리하면 한꺼번에 메모리에 올려야 하기 때문에 문제가 될 수 있다.
  * [[Dataset 클래스 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/datasets.py#L11)]
    * `__getitem__`과 `__len__` 메서드가 데이터셋의 핵심 메서드이다.
    * 데이터셋 전처리 : 데이터에서 특정 값을 제거하거나 데이터의 형상을 변형하는 처리
      * [`transforms.py`에 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/transforms.py)
      * 데이터 확장(data augmentation), 데이터 정규화(normalization) 등
  * Spiral 클래스
    * [[Spiral 클래스 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/datasets.py#L68)]
      * Dataset 클래스를 상속하고 `prepare`메서드에서 인스턴스 변수 data와 label에 데이터를 설정한다.
      * (또는 `__getitem__`이 불리는 시점에 데이터를 파일에서 읽어오는 방법도 있다.)
    * [Spiral 데이터셋 학습 예제](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/steps/step49.py)
      * 각 iteration에서 미니배치를 꺼내고(getitem), 그 후 ndarray 인스턴스로 변형한다.
* **DataLoader 클래스**
  * 미니배치 생성과 데이터셋 뒤섞기 등의 기능 제공
  * 반복자(iterator) : 원소를 반복하여 꺼내준다.
    * 파이썬의 `iter`함수를 이용하여 반복자로 변환하고, `next`함수를 이용하여 원소를 차례대로 꺼낸다.
    * 클래스에서 특수 메서드 구현을 통해 파이썬 반복자로 직접 만들 수도 있다.
      * `__iter__` : 자기 자신(self) 반환
      * `__next__` : 다음 원소 반환
  * [[DataLoader 클래스 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/dataloaders.py#L11)]
    * dataset : Dataset 인터페이스를 만족하는 인스턴스(`__getitem__`, `__len__` 메서드 구현됨)
    * batch_size : 배치 크기
    * shuffle : 에포크별로 데이터셋을 뒤섞을지 여부
* [Spiral 데이터셋 학습 예제](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/steps/step50.py)
  * (참고) [accuracy 함수](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/functions.py#L505) : 정답률 계산용, 계산 그래프 그리지 않음
* [MNIST 데이터셋 학습 예제](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/steps/step51.py)
  * (참고) [chainer 공식 mnist 예제](https://github.com/chainer/chainer/blob/master/examples/mnist/train_mnist_custom_loop.py)
  * (참고) [pytorch 공식 mnist 예제](https://github.com/pytorch/examples/blob/master/mnist/main.py)

