# 출처
> 밑바닥부터 시작하는 딥러닝 3 : https://www.hanbit.co.kr/store/books/look.php?p_code=B6627606922

> 소스코드 : https://github.com/WegraLee/deep-learning-from-scratch-3


# [DeZero] 5. 고급 기능


- 목차
    - [1. GPU 지원](#1-gpu-지원)
    - [2. 모델 저장 및 로드](#2-모델-저장-및-로드)
    - [3. 학습 및 추론 모드](#3-학습-및-추론-모드)
    - [4. CNN](#4-cnn)
    - [5. RNN](#5-rnn)
    - [6. 부록](#6-부록)


## 1. GPU 지원
### 환경 설정
* 엔비디아(NVIDIA)의 GPU, 쿠파이(CuPy) 라이브러리 필요
* 쿠파이(CuPy) : GPU를 활용하여 병렬 계산을 해주는 라이브러리
  * 설치 : `pip install cupy`
  * 장점
    * 넘파이와 API가 거의 같다.
    * 넘파이와 쿠파이의 다차원 배열을 서로 변환 가능
      ```python
      c = cp.asarray(n)
      n = cp.asnumpy(c)
      ```
    * 데이터에 적합한 모듈을 불러올 수 있어서, 쿠파이/넘파이 모두에 대응하는 코드를 작성할 수 있다.
      ```python
      xp = cp.get_array_module(x)
      y = xp.sin(x)
      ```
* 쿠다(CUDA) : 엔비디아가 제공하는 GPU용 개발 환경

### GPU 지원 구현
* 쿠다 모듈 구현
  * [`dezero/cuda.py`](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/cuda.py)
* 변수 관련 추가 구현
  * Variable
    * [`__init__`](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/core.py#L47) : 인수 data로 cupy.ndarray가 넘어와도 대응할 수 있도록 한다.
    * [`backward`](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/core.py#L93) : self.grad 변수 초기화 시, 데이터 타입에 따라 넘파이 또는 쿠파이 중 하나의 다차원 배열을 생성하도록 한다.
    * [`to_cpu`, `to_gpu`](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/core.py#L158) : Variable의 데이터를 CPU 또는 GPU로 전송한다.
  * Layer
    * [`to_cpu`, `to_gpu`](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/layers.py#L46) : Layer의 매개변수를 CPU 또는 GPU로 전송한다.
  * DataLoader
    * [`__init__`, `to_cpu`, `to_gpu`](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/dataloaders.py#L51) : gpu 플래그를 설정한다.
    * [`__next__`](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/dataloaders.py#L41) : gpu 플래그를 확인하여 쿠파이와 넘파이 중 알맞은 다차원 배열을 만들어준다.
* 함수 관련 추가 구현
  * [`dezero/functions.py`](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/functions.py) / [`dezero/layers.py`](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/layers.py) / [`dezero/optimizers.py`](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/optimizers.py)
    * 각 클래스에서 실제 계산은 `forward` 혹은 `update_one` 메서드에서 진행된다.
    * 이 메서드들에서는 ndarray 인스턴스를 사용한다고 가정하고 있었다.
    * 따라서 `np.`으로 시작하는 코드가 있는 부분에는 적합한 모듈을 불러오는 코드를 추가하여 변경한다.
    * (예시)
      ```python
      # before
      y = np.sin(x)
      # after
      xp = cuda.get_array_module(x)
      y = xp.sin(x)
      ```
  * [`dezero/core.py`](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/core.py)
    * [`as_array`](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/core.py#L177) : 인수에 array_module을 추가하여 numpy 또는 cupy 중 하나의 모듈의 ndarray로 변환해주도록 한다.
    * [사칙연산 함수](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/core.py#L210) : as_array 함수 사용 부분에 array_module 인수를 넣어서 호출한다.
* [GPU로 MNIST 학습 구현 예제](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/steps/step52.py)
  * (참고) 깃허브에 저장해둔 ipynb 파일은 `github.com`을 `colab.research.google.com/github`로 대체하면 구글 코랩에서 열 수 있다.
  * [GPU로 MNIST 학습 구현 예제(Colab)](https://colab.research.google.com/github/WegraLee/deep-learning-from-scratch-3/blob/master/examples/mnist_colab_gpu.ipynb)


## 2. 모델 저장 및 로드
* 모델 저장 및 로드 : 모델이 가지는 매개변수의 데이터를 외부 파일로 저장하고 다시 읽어오는 기능
  * 매개변수(Parameter 인스턴스) - 데이터(ndarray 인스턴스)
  * 일단은 넘파이의 ndarray를 저장하는 것만 고려
* 넘파이의 save 함수와 load 함수
  * `np.save('test.npy', x)` : ndarray 인스턴스 저장
  * `np.savez('test.npz', x1=x1, x2=x2)` : 여러 개의 ndarray 인스턴스 저장, 키워드 인수 지정 가능
  * `np.savez_compressed` : 내용 압축 기능 추가
  * `np.load` : 저장되어 있는 데이터를 읽어옴
  * (예시)
    ```python
    data = {'x1':x1, 'x2':x2}
    np.savez('test.npz', **data) # 저장
    arrays = np.load('test.npz') # 로드
    x1, x2 = arrays['x1'], arrays['x2']
    ```
* Layer 클래스의 save 함수와 load 함수
  * [`_flatten_params`](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/layers.py#L54) : Layer 내의 모든 Parameter를 평탄화하여 꺼낸다.
  * [`save_weights`](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/layers.py#L64) : 파일 저장, try 구문을 이용하여 불완전한 상태의 파일이 만들어지는 일을 사전에 방지한다.
  * [`load_weights`](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/layers.py#L78) : 데이터 로드
* [매개변수 저장 및 로드 예제](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/steps/step53.py)


## 3. 학습 및 추론 모드
### 과대 적합
* **과대적합(overfitting)** : 특정 훈련 데이터에 지나치게 최적화된 상태
1. 훈련 데이터가 적을 때
   * **데이터 확장(data augmentation)**
2. 모델의 표현력이 지나치게 높을 때
   * **가중치 감소(Weight Decay)**
   * **드롭아웃(Dropout)**
   * **배치 정규화(Batch Normalization)**

### 드롭아웃
* 드롭아웃(Dropout) : 뉴런을 임의로 삭제(비활성화)하면서 학습하는 방법, 앙상블 학습(Ensemble Learning)과 비슷한 효과
  * 학습 : 은닉층 뉴런을 무작위로 골라 삭제한다.
    ```python
    mask = np.random.rand(*x.shape) > dropout_ratio
    y = x * mask
    ```
  * 추론 : 모든 뉴런을 써서 출력을 계산하되, 그 결과를 약화시킨다.
    ```python
    scale = 1 - dropout_ratio
    y = x * scale
    ```
* 역 드롭아웃(Inverse Dropout) : 스케일 맞추기를 학습할 때 수행, 학습할 때 dropout_ratio를 동적으로 변경할 수 있음
  * 학습
    ```python
    scale = 1 - dropout_ratio
    mask = np.random.rand(*x.shape) > dropout_ratio
    y = x * mask / scale
    ```
  * 추론
    ```python
    y = x
    ```
* 드롭아웃 구현
  * [테스트 모드 추가](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/core.py#L29)
  * [드롭아웃 함수 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/functions.py#L517)
  * [드롭아웃 사용 예제](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/steps/step54.py)


## 4. CNN
### 합성곱 신경망(Convolutional Neural Network)
* **합성곱층(convolution layer)**
  * 합성곱 연산 : 입력 데이터에 대해 필터(filter or kernel)를 일정 간격으로 이동시키면서 적용한다. (+ 편향)
    * 필터가 가로와 세로 두 방향으로 이동하면 2차원 합성곱층이라고 한다. (Conv2d)
    * 패딩(padding) : 출력 크기를 조정하기 위해, 주요 처리 전에 입력 데이터 주위에 고정값(0 등)을 채우는 것
    * 스트라이드(stride) : 필터를 적용하는 위치의 '간격'
    * [출력 크기 계산](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/utils.py#L380)
  * 3차원 텐서에 대한 합성곱 연산
    * 데이터 : (채널(channel), 높이(height), 너비(width)) 순으로 정렬
    * 출력 : 특징 맵(feature map), 다수의 필터(가중치)를 사용하면 특징 맵이 여러 장 출력된다.
  * 4차원 텐서(미니배치 처리)
    * 데이터 : (미니배치(batch_size), 채널(channel), 높이(height), 너비(width)) 형상으로 정렬
  * Conv2d 함수 구현
    * [`im2col`](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/functions_conv.py#L303) : 입력 데이터를 한 줄로 전개하는 함수(3차원 텐서->2차원 텐서(행렬))
      * 입력 데이터를 한줄로 전개한 후 합성곱층의 커널도 한 줄로 전개하여 두 행렬을 곱하기 위함
      * (참고) 텐서 곱 : 행렬 곱의 확장, 곱셈-누적 연산(multiply-accumulate operation)
    * [`pair`](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/utils.py#L384) : 원소 2개짜리 튜플 반환
    * [[기본 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/functions_conv.py#L11)]
      * 순전파 : im2col 함수 이용, 행렬 곱 방식으로 계산
    * [[Function으로 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/functions_conv.py#L49)]
      * 순전파 : im2col 함수 이용, 텐서 곱 방식으로 계산
      * 역전파 : 전치 합성곱(transposed convolution) 방식으로 계산
  * Conv2d 계층 구현
    * [[Layer로 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/layers.py#L120)]
* **풀링층(pooling layer)**
  * 풀링 : 가로, 세로 공간을 작게 만드는 연산
    * Max 풀링 : 대상 영역에서 최댓값을 취한다. (주로 사용)
    * Average 풀링 : 대상 영역의 평균을 계산한다.
  * 특징
    * 학습하는 매개변수가 없다.
    * 채널 수가 변하지 않는다.
    * 미세한 위치 변화에 영향을 덜 받는다.(robust)
  * pooling 함수 구현
    * [`max`](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/functions.py#L609) : 인수 axis를 지정하여 어떤 축을 기준으로 최댓값을 구할지 명시할 수 있음
    * [[기본 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/functions_conv.py#L29)]
      * 순전파 : im2col 함수 이용, 풀링의 적용 영역은 채널마다 독립적으로 전개

### 대표적인 CNN 구현
* **VGG16**
  * 2014년 ILSVRC 대회에서 준우승한 모델
  * 특징
    * 3x3 합성곱층 사용(1x1 패딩)
    * 풀링 시 채널 수 2배 증가
    * 완전연결계층에서 드롭아웃 사용
    * 활성화 함수 ReLU 사용
  * [[Model로 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/models.py#L52)]
    * `pretrained` : 학습된 가중치 데이터를 읽어오는 기능을 위한 플래그
  * [이미지 분류 예제](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/steps/step58.py)
* **ResNet**
  * [[Model로 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/models.py#L118)]


## 5. RNN
### 피드포워드 구조의 신경망(feed-forward Neural Network)
* 특징
  * 데이터를 순방향으로만 계속 입력해준다.
  * 즉, 입력 신호만으로 출력을 결정한다.

### 순환 신경망(Recurrent Neural Network)
* 특징
  * 순환(loop) 구조를 가지고 있다.
  * 즉, 데이터가 입력되면 '상태'가 갱신되고 그 '상태'도 출력에 영향을 준다.
  * 입력 데이터가 나열되는 패턴을 학습할 수 있다.
* ![image](https://user-images.githubusercontent.com/35680202/129688643-5a8da4eb-680d-4997-829c-1146bfa146fe.png)
* RNN 계층 구현
  * [[Layer로 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/layers.py#L221)]
    * x2h : 입력 x에서 은닉 상태 h로 변환하는 완전연결계층
    * h2h : 이전 은닉 상태에서 다음 은닉 상태로 변환하는 완전연결계층 (편향 생략)
    * `forward` : 새로운 은닉 상태를 계산하여 반환
* RNN 모델 구현
  * [SimpleRNN 구현 예제](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/steps/step59.py)
    * [사인파(sine wave)](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/datasets.py#L260) 예측
    * BPTT(Backpropagation Through Time) : 일련의 입력 데이터로 구성된 계산 그래프에서의 역전파
    * Truncated BPTT : 역전파를 잘하려면 계산 그래프를 적당한 길이에서 끊어줘야 한다. 은닉 상태는 유지된다.
      * [`unchain`](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/core.py#L85) : 연결을 끊어주는 메서드, 창조자인 self.creator를 None으로 설정한다.
      * [`unchain_backward`](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/core.py#L128) : 변수와 함수를 거꾸로 거슬러 올라가면서 변수의 `unchain` 메서드를 호출하여 연결을 끊는다.

### RNN 구현 개선
* 시계열 데이터용 데이터 로더 : 여러 데이터를 묶은 미니배치 단위로 순전파할 수 있도록 함
  * [[DataLoader로 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/dataloaders.py#L58)]
    * 데이터의 순서가 바뀌면 안 된다.
    * batch_index 위치에서 시작해서 batch_size 개수 만큼 가져온다.
* LSTM 계층
  * ![image](https://user-images.githubusercontent.com/35680202/129689345-ec15915b-bbef-419e-8b84-56a097e81d1a.png)
  * [[Layer로 구현](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/layers.py#L249)]
* 개선된 RNN 모델 구현
  * [BetterRNN 구현 예제](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/steps/step60.py)
    * 사인파 예측
    * 시계열 데이터용 데이터 로더 사용
    * LSTM 계층 사용


## 6. 부록
* 정적 계산 그래프와 ONNX 지원
  * ONNX(Open Neural Network Exchange) : 딥러닝 모델을 표현하는데 사용되는 포맷, 학습된 모델을 다른 프레임워크로 쉽게 이식
* PyPI에 공개
  * 파이썬 패키지 저장소
  * `pip install`로 설치
* API 문서 준비
  * docstring : 파이썬 문서화용 주석, 개발자가 함수나 클래스 등을 설명하는 용도로 이용
  * Sphinx : 코드에 docstring이 준비되어 있을 때, Sphinx와 같은 도구를 사용하여 HTML이나 PDF 등의 포맷으로 API 문서를 뽑을 수 있음
* 로고 제작
* 구현 예 추가

