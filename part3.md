# 출처
> 밑바닥부터 시작하는 딥러닝 3 : https://www.hanbit.co.kr/store/books/look.php?p_code=B6627606922

> 소스코드 : https://github.com/WegraLee/deep-learning-from-scratch-3


# [DeZero] 3. 고차 미분 계산
* 고차 미분 : 어떤 함수를 2번 이상 미분한 것

- 목차
    - [1. 계산 그래프 시각화](#1-계산-그래프-시각화)
    - [2. 테일러 급수 미분](#2-테일러-급수-미분)
    - [3. 함수 최적화](#3-함수-최적화)
    - [4. 고차 미분](#4-고차-미분)
    - [5. 함수 구현](#5-함수-구현)
    - [6. 부록](#6-부록)


## 1. 계산 그래프 시각화
### Graphviz와 DOT 언어
* Graphviz : 그래프(노드와 화살표로 이뤄진 데이터 구조)를 시각화해주는 도구
* Graphviz 설치하기
  1. [Windows용 다운로드 및 설치](https://graphviz.org/download/#windows)
  2. 환경변수 `path`에 `C:\Program Files\Graphviz\bin`를 추가해준다.
  3. 아나콘다 프롬프트에서 graphviz를 설치 후 dot 명령을 실행해본다.
     ```bash
     conda install python-graphviz
     dot -V
     ```
* DOT 언어 : 간단한 문법으로 그래프를 작성할 수 있다.
  * 기본 구조 : `digraph g {...}`
  * 각 노드를 줄바꿈으로 구분
  * 노드ID는 0 이상의 정수이며, 다른 노드와 중복 불가능
  * 예시 코드 (결과 : `x` -> `Exp` -> `y`)
    ```dot
    digraph g{
    1 [label='x', color=orange, style=filled]
    2 [label='y', color=orange, style=filled]
    3 [label='Exp', color=lightblue, style=filled, shape=box]
    1 -> 3
    3 -> 2
    }
    ```
  * 이미지로 변환 : `dot sample.dot -T png -o smaple.png`

### 계산 그래프 시각화 코드
* [`dezero/utils.py`](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/utils.py#L11)
  * `get_dot_graph`에서만 사용하는 보조 함수는 이름 앞에 밑줄(`_`)을 붙였다.
  * `id()` : 파이썬 내장함수로, 주어진 객체의 ID를 반환한다. 고유한 노드ID로 사용하기에 적합하다.
  * `get_dot_graph`는 Variable.backward() 메서드와 거의 비슷한 흐름으로 구현한다.
* [계산 그래프 시각화 예시](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/steps/step26.py)


## 2. 테일러 급수 미분
### 테일러 급수 이론
* 테일러 급수(Taylor Series) : 어떤 함수를 다항식으로 근사하는 방법
  * <!-- $f(x) = f(a) + f'(a)(x - a) + \frac{1}{2!}f''(a)(x - a)^2 + \frac{1}{3!}f'''(a)(x - a)^3 + ...$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=f(x)%20%3D%20f(a)%20%2B%20f'(a)(x%20-%20a)%20%2B%20%5Cfrac%7B1%7D%7B2!%7Df''(a)(x%20-%20a)%5E2%20%2B%20%5Cfrac%7B1%7D%7B3!%7Df'''(a)(x%20-%20a)%5E3%20%2B%20...">
  * : 점 <!-- $a$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=a">에서 <!-- $f(x)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=f(x)">의 테일러 급수, 항이 많아질수록 근사의 정확도가 높아짐
* 매클로린 전개(Maclaurin's series) : <!-- $a = 0$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=a%20%3D%200">일 때의 테일러 급수
  * <!-- $f(x) = f(0) + f'(0)x + \frac{1}{2!}f''(0)x^2 + \frac{1}{3!}f'''(0)x^3 + ...$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=f(x)%20%3D%20f(0)%20%2B%20f'(0)x%20%2B%20%5Cfrac%7B1%7D%7B2!%7Df''(0)x%5E2%20%2B%20%5Cfrac%7B1%7D%7B3!%7Df'''(0)x%5E3%20%2B%20...">
* [테일러 급수 구현 예제](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/steps/step27.py)
  * <!-- $sin(x) = \frac{x}{1!} - \frac{x^3}{3!} + \frac{x^5}{5!} + ... = \sum_{i=0}^{\infty} (-1)^{i} \frac{x^{2i + 1}}{(2i + 1)!}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=sin(x)%20%3D%20%5Cfrac%7Bx%7D%7B1!%7D%20-%20%5Cfrac%7Bx%5E3%7D%7B3!%7D%20%2B%20%5Cfrac%7Bx%5E5%7D%7B5!%7D%20%2B%20...%20%3D%20%5Csum_%7Bi%3D0%7D%5E%7B%5Cinfty%7D%20(-1)%5E%7Bi%7D%20%5Cfrac%7Bx%5E%7B2i%20%2B%201%7D%7D%7B(2i%20%2B%201)!%7D">
  * 테일러 급수의 임곗값(threshhold)을 작게할 수록 이론상으로 근사 정밀도가 높아진다. 하지만 컴퓨터 계산에서는 자릿수 누락이나 반올림이 발생할 수 있다.


## 3. 함수 최적화
### 최적화
* 최적화 : 어떤 함수가 주어졌을 때, 그 최솟값(혹은 최댓값)을 반환하는 '입력(함수의 인수)'을 찾는 일, 신경망 학습의 목표는 손실 함수의 최적화이다.
* 함수 예시
  * 로젠브록 함수(Rosenbrock function, Banana function)
    * <!-- $f(x_0, x_1) = b(x_1 - x_0^2)^2 + (a - x_0)^2$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=f(x_0%2C%20x_1)%20%3D%20b(x_1%20-%20x_0%5E2)%5E2%20%2B%20(a%20-%20x_0)%5E2"> (단, <!-- $a,b$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=a%2Cb">는 정수)
    * 벤치마크로 자주 쓰이는 이유 : 골짜기로 향하는 기울기에 비해 골짜기 바닥에서 전역 최솟값으로 가는 기울기가 너무 작아서 최적화하기가 어렵기 때문

### 경사하강법
* 기울기(gradient) : 각 지점에서 함수의 출력을 (적어도 국소적으로는) 가장 크게(+)/작게(-) 하는 방향을 가리킨다.
* 경사하강법(gradient descent) : 기울기 방향에 마이너스를 곱한 방향으로 일정 거리만큼 이동하여 다시 기울기를 구하는 작업을 반복하여 원하는 지점에 접근하는 방법
  * <!-- $x \leftarrow x - \alpha f'(x)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x%20%5Cleftarrow%20x%20-%20%5Calpha%20f'(x)">
  * <!-- $\alpha$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Calpha">는 수동 설정, <!-- $\alpha$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Calpha">의 값만큼 기울기(1차 미분) 방향으로 진행하여 <!-- $x$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x">의 값을 갱신한다.
* [경사하강법 최적화 예시](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/steps/step28.py)

### 뉴턴 방법
* 2차까지 테일러 급수로 근사
  * 어떤 함수를 <!-- $x$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x">의 2차 함수로 근사한 것을 <!-- $y = f(x)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=y%20%3D%20f(x)">라고 하자.
    * <!-- $f(x) = f(a) + f'(a)(x - a) + \frac{1}{2} f''(a)(x - a)^2$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=f(x)%20%3D%20f(a)%20%2B%20f'(a)(x%20-%20a)%20%2B%20%5Cfrac%7B1%7D%7B2%7D%20f''(a)(x%20-%20a)%5E2">
  * 근사한 2차 함수의 최솟값은 2차 함수의 미분 결과가 0인 위치에 있다.
    * <!-- $x = a - \frac{f'(a)}{f''(a)}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x%20%3D%20a%20-%20%5Cfrac%7Bf'(a)%7D%7Bf''(a)%7D">
* 뉴턴 방법(Newton's method) : 2차 미분을 이용하여 <!-- $\alpha$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Calpha">를 자동으로 조정한다.
  * <!-- $x \leftarrow x - \frac{f'(x)}{f''(x)}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x%20%5Cleftarrow%20x%20-%20%5Cfrac%7Bf'(x)%7D%7Bf''(x)%7D">
  * <!-- $\alpha = \frac{1}{f''(x)}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Calpha%20%3D%20%5Cfrac%7B1%7D%7Bf''(x)%7D">
* [뉴턴 방법 최적화 예시](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/steps/step29.py)


## 4. 고차 미분
### 목표
* [고차 미분 구현을 통해 뉴턴 방법을 활용한 최적화 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/steps/step33.py)

### 방법
* 계산 그래프의 **연결**은 Function 클래스의 `__call__`메서드에서 **순전파**를 계산할 때 만들어진다.
* **역전파**를 계산할 때도 **연결**을 만들 수 있다면, 고차 미분 구현 가능
* 따라서 **미분값(기울기)을 Variable 인스턴스로** 만들기
  1. output변수의 grad를 Variable 인스턴스로 만든다.
  2. Function의 backward에서 계산하는 변수도 Variable 그대로 사용한다.
  3. 그러면 다음 세대에서도 grad가 계속 Variable 인스턴스가 된다.
* 역전파 활성/비활성 모드(create_graph)
  * 역전파 활성화
    * (Function의 `backward`메서드->(다른 Function 인스턴스의)`__call__`메서드) 과정에서 그 다음 역전파를 위한 변수를 저장하고, 연결을 만든다.
    * 따라서 미분값을 계산하는 과정에서의 계산 그래프를 그린다. (고차 미분 가능)
    * `y.backward(create_graph=True)`으로 사용한다.
  * 역전파 비활성화
    * (Function의 `backward`메서드->(다른 Function 인스턴스의)`__call__`메서드) 과정에서 그 다음 역전파를 위한 변수는 저장하지 않고, 연결도 만들지 않는다.
    * 따라서 미분값을 계산하는 과정에서의 계산 그래프를 그리지 않는다. (고차 미분 필요 없음)
    * `gx.backward()`으로 사용한다.
* 미분값 재설정
  * 문제 : 미분값 누적
    ```python
    y.backward(create_graph=True)
    print(x.grad) # x.grad = dy/dx
    gx = x.grad
    gx.backward()
    print(x.grad) # x.grad = dy/dx + dy/d^2x
    ```
  * 해결 : 미분값 재설정
    ```python
    y.backward(create_graph=True)
    print(x.grad) # x.grad = dy/dx
    gx = x.grad
    x.cleargrad() # 미분값 재설정(x.grad = None)
    gx.backward()
    print(x.grad) # x.grad = dy/d^2x
    ```

### 고차 미분 구현
* [`dezero/core.py`](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/core.py)
```python
class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    
    def cleargrad(self):
      self.grad = None
    
    def backward(self, retain_grad=False, create_graph=False): 
        if self.grad is None:
            #self.grad = np.ones_like(self.data)
            self.grad = Variable(np.ones_like(self.data)) # grad를 Variable 인스턴스로 만들기

        funcs = []
        seen_set = set()
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x : x.generation)

        add_func(self.creator)
        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]

            with using_config('enable_backprop', create_graph): # 역전파 활성/비활성 모드 전환
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)
            
                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx
                    if x.creater is not None:
                        add_func(x.creator)
            
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

# ===========================================================
class Config:
    enable_backprop = True

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

# ===========================================================
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop: # 역전파 활성/비활성 모드 제어 부분
            for output in outputs:
                output.set_creator(self) # 계산 그래프 연결
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
            self.generation = max([x.generation for x in inputs])
        
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()
    
    def backward(self, gys):
        raise NotImplementedError()

# === 연산자 오버로드 =======================================
class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    def backward(self, gy):
        #x0, x1 = self.inputs[0].data, self.inputs[1].data
        x0, x1 = self.inputs # Variable 인스턴스로 계산하여 연결 만들기
        return gy * x1, gy * x0
```


## 5. 함수 구현
* [`dezero/functions.py`](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/functions.py#L8)
  * Function의 backward 메서드 안의 모든 변수는 Variable 인스턴스이다.
  * 따라서 backward 메서드 구현 시 모든 계산은 반드시 DeZero 함수를 사용해야 한다.
* [함수 고차 미분 예시](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/steps/step34.py) / [함수 고차 미분 계산 그래프 예시](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/steps/step35.py) / [고해상도 예시](https://github.com/WegraLee/deep-learning-from-scratch-3/tree/tanh)


## 6. 부록
### 뉴턴 방법의 한계와 대안
* 다변수 함수의 뉴턴 방법 (즉, <!-- $x = (x_1, x_2, ..., x_n)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x%20%3D%20(x_1%2C%20x_2%2C%20...%2C%20x_n)">일 때, <!-- $y = f(x)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=y%20%3D%20f(x)">에 대한 뉴턴 방법)
  * <!-- $x \leftarrow x - [{\nabla}^2 f(x)]^{-1} \nabla f(x)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x%20%5Cleftarrow%20x%20-%20%5B%7B%5Cnabla%7D%5E2%20f(x)%5D%5E%7B-1%7D%20%5Cnabla%20f(x)">
  * <!-- $\nabla f(x)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cnabla%20f(x)"> : 기울기(gradient), <!-- $x$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x">의 각 원소에 대한 미분
    * <!-- $\nabla f(x) = \frac{\partial f}{\partial x} = (\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n})^{T}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cnabla%20f(x)%20%3D%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x%7D%20%3D%20(%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_1%7D%2C%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_2%7D%2C%20...%2C%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_n%7D)%5E%7BT%7D">
  * <!-- ${\nabla}^2 f(x)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%7B%5Cnabla%7D%5E2%20f(x)"> : 헤세 행렬(Hessian matrix)
    * <!-- ${\nabla}^2 f(x) = \frac{{\partial}^2 f}{\partial x \partial x^T} = \begin{pmatrix} \frac{{\partial}^2 f}{\partial x_1^2} & \frac{{\partial}^2 f}{\partial x_1 \partial x_2} & ... & \frac{{\partial}^2 f}{\partial x_1 \partial x_n} \\ \frac{{\partial}^2 f}{\partial x_2 \partial x_1} & \frac{{\partial}^2 f}{\partial x_2^2} & ... & \frac{{\partial}^2 f}{\partial x_2 \partial x_n} \\ ... & ... & ... & ... \\ \frac{{\partial}^2 f}{\partial x_n \partial x_1} & \frac{{\partial}^2 f}{\partial x_n \partial x_2} & ... & \frac{{\partial}^2 f}{\partial x_n^2} \end{pmatrix}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%7B%5Cnabla%7D%5E2%20f(x)%20%3D%20%5Cfrac%7B%7B%5Cpartial%7D%5E2%20f%7D%7B%5Cpartial%20x%20%5Cpartial%20x%5ET%7D%20%3D%20%5Cbegin%7Bpmatrix%7D%20%5Cfrac%7B%7B%5Cpartial%7D%5E2%20f%7D%7B%5Cpartial%20x_1%5E2%7D%20%26%20%5Cfrac%7B%7B%5Cpartial%7D%5E2%20f%7D%7B%5Cpartial%20x_1%20%5Cpartial%20x_2%7D%20%26%20...%20%26%20%5Cfrac%7B%7B%5Cpartial%7D%5E2%20f%7D%7B%5Cpartial%20x_1%20%5Cpartial%20x_n%7D%20%5C%5C%20%5Cfrac%7B%7B%5Cpartial%7D%5E2%20f%7D%7B%5Cpartial%20x_2%20%5Cpartial%20x_1%7D%20%26%20%5Cfrac%7B%7B%5Cpartial%7D%5E2%20f%7D%7B%5Cpartial%20x_2%5E2%7D%20%26%20...%20%26%20%5Cfrac%7B%7B%5Cpartial%7D%5E2%20f%7D%7B%5Cpartial%20x_2%20%5Cpartial%20x_n%7D%20%5C%5C%20...%20%26%20...%20%26%20...%20%26%20...%20%5C%5C%20%5Cfrac%7B%7B%5Cpartial%7D%5E2%20f%7D%7B%5Cpartial%20x_n%20%5Cpartial%20x_1%7D%20%26%20%5Cfrac%7B%7B%5Cpartial%7D%5E2%20f%7D%7B%5Cpartial%20x_n%20%5Cpartial%20x_2%7D%20%26%20...%20%26%20%5Cfrac%7B%7B%5Cpartial%7D%5E2%20f%7D%7B%5Cpartial%20x_n%5E2%7D%20%5Cend%7Bpmatrix%7D">
  * <!-- $x$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x">를 기울기 방향으로 갱신하고, 그 진행 거리를 헤세 행렬의 역행렬을 사용하여 조정한다.
* 한계 : 매개변수가 많아지면 헤세 행렬의 역행렬 계산에 너무 많은 자원이 소모된다.
  * 매개변수 <!-- $n$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=n">개면 메모리 <!-- $n^2$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=n%5E2">만큼 사용, <!-- $n \times n$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=n%20%5Ctimes%20n">의 역행렬 계산에는 <!-- $n^3$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=n%5E3">만큼 사용
* 대안
  * 준 뉴턴 방법(Quasi-Newton Method, QNM) : 뉴턴 방법 중 헤세 행렬의 역행렬을 근사하여 사용하는 방법의 총칭
    * ex) L-BFGS 등
  * 기울기만을 사용한 최적화
    * ex) SGD, Momentum, Adam 등

### double backprop의 다양한 용도
* double backpropagation : 역전파를 수행한 계산에 대해 또 다시 역전파하는 것
* 용도
  * 고차 미분
  * [미분이 포함된 식에서의 미분](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/steps/step36.py)
    * 예시 : [WGAN-GP에서 최적화하는 함수 $L$](https://paperswithcode.com/method/wgan-gp-loss)에 기울기가 들어있다. 함수 $L$을 최적화하기 위해 두번째 역전파를 한다.
  * 헤세 행렬과 벡터의 곱(Hessian-vector product)
    * <!-- ${\nabla}^2 f(x) v = \nabla (v^T \nabla f(x))$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%7B%5Cnabla%7D%5E2%20f(x)%20v%20%3D%20%5Cnabla%20(v%5ET%20%5Cnabla%20f(x))">
    * 오른쪽처럼, 벡터의 내적을 먼저 구하고 그 결과로부터 다시 한 번 기울기를 구함으로써, 해세 행렬을 만들지 않고도 값을 구할 수 있다.
    * 예시 : [TRPO(Trust Region Policy Optimization)](https://arxiv.org/abs/1502.05477)에서는 헤세 행렬과 벡터의 곱을 구할 때 double backprop을 사용한다.
      * ![image](https://user-images.githubusercontent.com/35680202/129677014-d36fb5b4-8bc3-46c5-9efc-77c5fcddad2e.png)
      * ![image](https://user-images.githubusercontent.com/35680202/129677062-7ba3401d-3ce5-4935-86e6-d62fabca0f5e.png)
