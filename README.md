# 악성 URL 체커 (Malicious URL Checker)

AutoGluon과 BERT 기반 모델을 사용하여 악성 URL을 탐지하는 머신러닝 시스템입니다.

## 주요 기능

- 🔍 URL 악성 여부 실시간 예측
- 📊 대량 URL 일괄 분석
- 🧠 BERT 기반 딥러닝 모델 사용
- 📈 URL 특징 분석 및 시각화
- 🚀 GPU 가속 지원

## 설치

### 1. 환경 설정

```bash
# Python 3.11+ 필요
python --version

# UV 패키지 매니저 설치 (선택사항)
pip install uv

# 가상환경 생성 및 활성화
uv venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 2. 의존성 설치

```bash
# UV 사용시
uv add -r requirements.txt

# 또는 pip 사용시
pip install -r requirements.txt
```

## 데이터 준비

### 1. 정상 URL 데이터 생성

```bash
python generate_normal_urls.py
```

### 2. 데이터 구조

```
data/
├── url_1.csv       # 스팸/피싱 URL 데이터 (hxxp 형식)
├── url_2.csv       # 추가 악성 URL 데이터
└── normal_urls.csv # 생성된 정상 URL 데이터
```

## 모델 학습

### 기본 학습 (CPU/간단한 GPU)

```bash
python main.py
```

### 고성능 GPU 학습 (A6000 등)

```bash
python train_gpu.py
```

학습 옵션:
- 시간 제한: `time_limit` 파라미터 조정 (기본 600초)
- 에포크 수: `optimization.max_epochs` 조정
- 배치 크기: `optimization.batch_size` 조정

## URL 예측

### 1. 대화형 모드

```bash
python predict_urls.py
```

### 2. 단일 URL 예측

```bash
python predict_urls.py --url "https://suspicious-site.com"
```

### 3. 파일에서 일괄 예측

```bash
# CSV 또는 텍스트 파일에서 URL 읽기
python predict_urls.py --file urls.txt --output results.json
```

### 4. 모델 경로 지정

```bash
python predict_urls.py --model ./malicious_url_model_gpu --url "https://example.com"
```

## 사용 예시

### Python 코드에서 사용

```python
from main import MaliciousURLChecker

# 체커 초기화
checker = MaliciousURLChecker()

# 모델 로드 (이미 학습된 경우)
checker.load_model()

# URL 예측
urls = ["https://bit.ly/suspicious", "https://google.com"]
results = checker.predict_urls(urls)

for result in results:
    print(f"URL: {result['url']}")
    print(f"악성 여부: {result['is_malicious']}")
    print(f"악성 확률: {result['malicious_probability']:.2%}")
```

## 모델 성능

학습된 모델은 다음과 같은 지표로 평가됩니다:
- **ROC-AUC**: 악성/정상 분류 성능
- **Accuracy**: 전체 정확도
- **Precision**: 악성으로 예측한 URL 중 실제 악성 비율
- **Recall**: 실제 악성 URL 중 탐지한 비율
- **F1-Score**: Precision과 Recall의 조화평균

## 프로젝트 구조

```
malicious-url-checker/
├── data/                    # 데이터 디렉토리
│   ├── url_1.csv           # 악성 URL 데이터 1
│   ├── url_2.csv           # 악성 URL 데이터 2
│   └── normal_urls.csv     # 정상 URL 데이터
├── main.py                 # 메인 학습 스크립트
├── train_gpu.py           # GPU 최적화 학습 스크립트
├── predict_urls.py        # URL 예측 스크립트
├── generate_normal_urls.py # 정상 URL 생성 스크립트
├── malicious_url_model/   # 학습된 모델 저장 디렉토리
├── pyproject.toml         # 프로젝트 설정
├── uv.lock               # 의존성 잠금 파일
└── README.md             # 이 파일
```

## 주의사항

1. **데이터 불균형**: 악성 URL이 정상 URL보다 많은 경우 추가적인 정상 URL 데이터가 필요합니다.
2. **메모리 사용**: 대용량 데이터셋 학습 시 충분한 RAM이 필요합니다 (최소 16GB 권장).
3. **GPU 메모리**: GPU 학습 시 배치 크기를 GPU 메모리에 맞게 조정하세요.
4. **URL 정규화**: 입력 URL은 자동으로 정규화됩니다 (hxxp → http).

## 기여하기

버그 리포트, 기능 제안, 풀 리퀘스트를 환영합니다!

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.
