import pandas as pd
import os
from main import MaliciousURLChecker

def test_small_dataset():
    """작은 데이터셋으로 빠른 테스트"""
    print("작은 데이터셋으로 테스트를 시작합니다...")
    
    # 체커 초기화
    checker = MaliciousURLChecker(model_path='./test_model')
    
    # 데이터 로드
    df = checker.load_and_preprocess_data()
    
    # 매우 작은 샘플만 사용 (빠른 테스트를 위해)
    sample_size = min(100, len(df))
    df_small = df.sample(n=sample_size, random_state=42)
    
    print(f"\n테스트용 샘플 크기: {len(df_small)}")
    print(f"악성 URL: {len(df_small[df_small['label']==1])}")
    print(f"정상 URL: {len(df_small[df_small['label']==0])}")
    
    # 학습 데이터 준비
    train_df, test_df, class_weights = checker.prepare_training_data(df_small, test_size=0.3)
    
    # 빠른 학습 (시간 제한 60초)
    print("\n빠른 테스트 학습을 시작합니다 (60초 제한)...")
    
    # 간단한 하이퍼파라미터로 재정의
    checker.predictor = None
    from autogluon.multimodal import MultiModalPredictor
    
    checker.predictor = MultiModalPredictor(
        label='label',
        problem_type='binary',
        eval_metric='accuracy',
        path=checker.model_path
    )
    
    # 매우 간단한 설정으로 학습
    hyperparameters = {
        'model.hf_text.checkpoint_name': 'distilbert-base-uncased',  # 가벼운 모델
        'optimization.max_epochs': 1,
        'optimization.batch_size': 8,
    }
    
    checker.predictor.fit(
        train_data=train_df,
        hyperparameters=hyperparameters,
        time_limit=60,  # 1분 제한
        presets='medium_quality'  # 빠른 프리셋
    )
    
    print("테스트 학습 완료!")
    
    # 간단한 평가
    scores = checker.predictor.evaluate(test_df, metrics=['accuracy'])
    print(f"\n테스트 정확도: {scores['accuracy']:.4f}")
    
    # 예측 테스트
    print("\n=== 예측 테스트 ===")
    test_urls = [
        'https://bit.ly/suspicious',
        'https://www.google.com',
        'http://192.168.1.1/admin',
        'https://github.com'
    ]
    
    results = checker.predict_urls(test_urls)
    for result in results:
        status = "악성" if result['is_malicious'] else "정상"
        print(f"URL: {result['url']}")
        print(f"  상태: {status}")
        print(f"  악성 확률: {result['malicious_probability']:.2%}\n")

if __name__ == "__main__":
    test_small_dataset() 