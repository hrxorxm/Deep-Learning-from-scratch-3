# 출처
> 밑바닥부터 시작하는 딥러닝 3 : https://www.hanbit.co.kr/store/books/look.php?p_code=B6627606922

> 소스코드 : https://github.com/WegraLee/deep-learning-from-scratch-3


# [DeZero] 1. 미분 자동 계산(Automatic Differential Calculation)

- 목차
    - [1. 변수](#1-변수)
    - [2. 함수](#2-함수)
    - [3. 미분](#3-미분)
    - [4. 구현](#4-구현)
    - [5. 테스트](#5-테스트)


## 1. 변수
### 변수(Variable)
* 변수와 데이터는 별개다.
* 변수에 데이터를 대입 혹은 할당한다.
* 변수를 들여다보면 데이터를 알 수 있다(참조한다).

### 다차원 배열(Tensor)
* 차원(dimension), 축(axis) : 다차원 배열에서 원소의 순서의 방향
* 스칼라(scalar) : 0차원 배열
* 벡터(vector) : 1차원 배열
* 행렬(matrix) : 2차원 배열


## 2. 함수
### 함수(Function)
* 정의 : 어떤 변수로부터 다른 변수로의 대응 관계를 정한 것
* 합성 함수(composite function) : 여러 함수로 구성된 함수
* 계산 그래프(computational graph) : 노드들을 화살표로 연결해 계산 과정을 표현한 그림, 계산 그래프를 이용하면 각 변수에 대한 미분을 효율적으로 계산할 수 있다.


## 3. 미분
### 수치 미분
* 미분 : (극한으로 짧은 시간에서의) 변화율
* 전진차분(forward difference)
  * $f'(x) = \underset{h \rightarrow 0}{\lim} \frac{f(x+h) - f(x)}{h}$
* 중앙차분(centered difference)
  * $f'(x) = \underset{h \rightarrow 0}{\lim} \frac{f(x+h) - f(x-h)}{2h}$
  * 근사 오차를 (조금이라도) 줄이는 방법
* 수치 미분(numerical differentation) : 근사값를 이용하여 함수의 변화량을 구하는 방법
  * (예시) $h \rightarrow 0$ 대신 $h = 1e-4$ 를 대입해서 계산
  * 장점 : 쉬운 구현
  * 단점 : 큰 계산 비용, 정확도 문제(자릿수 누락)
* 기울기 확인(gradient checking) : 역전파를 정확하게 구현했는지 확인하기 위해 수치 미분의 결과 이용

### 역전파
* 역전파(backpropagation) : 변수별 미분을 계산하는 알고리즘
* 연쇄 법칙(chain rule) : 합성 함수의 미분은 구성 함수 각각을 미분한 후 곱한 것과 같다.
  * $a = A(x), b = B(a), y = C(b)$ 일 때,
  * $\frac{dy}{dx} = ((\frac{dy}{dy} \frac{dy}{db}) \frac{db}{da}) \frac{da}{dx}$
* 손실 함수(loss function)의 각 매개변수에 대한 미분을 계산하기 위해 사용
* 순전파와 역전파의 대응관계
  * 통상값, 통상 계산(순전파)
  * 미분값, 미분값을 구하기 위한 계산(역전파)
  * $C'(b)$ 를 계산하기 위해서 $b$ 값이 필요하다.
  * 역전파를 구현하기 위해서는 먼저 순전파를 하고, 각 함수의 입력 변수의 값을 기억해둬야 한다.
* 역전파 자동화
  * Define-by-Run
  * Wengert List(or tape)
  * 동적 계산 그래프(Dynamic Computational Graph)


## 4. 구현
```python
import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))

        self.data = data # 통상값
        self.grad = None # 미분값
        self.creator = None # 함수와의 관계 (핵심☆)
    
    def set_creator(self, func):
        self.creator = func
    
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data) # dy/dy = 1

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)

def as_array(x):
    '''0차원 ndarray 인스턴스는 계산 결과의 데이터 타입이 달라지기 때문에 조정해야 한다.'''
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self) # 함수와의 관계 설정
        self.input = input # 입력 변수 기억
        self.output = output # 출력 변수 기억
        return output

    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()

# ================ 예제 ================================
# 예제 계산 클래스 정의
class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

# 파이썬 함수로 정의
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

# 예제 실행
x = Variable(np.array(0.5))
y = square(exp(square(x))) # 합성 함수
y.backward() # 맨 마지막 변수만 backward 호출

print(x.grad) # 3.297442541400256
```


## 5. 테스트
* [`test.py`](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/steps/step10.py#L78)
* 실행 방법
  * add to last line in test.py
    ```python
    unittest.main()
    ```
    and run
    ```python
    python test.py
    ```
  * run test.py
    ```bash
    python -m unittest test.py
    ```
  * run tests/test*.py
    ```bash
    python -m unittest discover tests
    ```