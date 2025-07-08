import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from autogluon.multimodal import MultiModalPredictor
import warnings
warnings.filterwarnings('ignore')

class MaliciousURLChecker:
    def __init__(self, data_dir='data', model_path='./malicious_url_model'):
        self.data_dir = data_dir
        self.model_path = model_path
        self.predictor = None
        
    def load_and_preprocess_data(self):
        """데이터 로드 및 전처리"""
        print("데이터를 로드하고 있습니다...")
        
        # URL 1 데이터 로드 (스팸 URL)
        url1_path = os.path.join(self.data_dir, 'url_1.csv')
        print(f"url_1.csv 로드 중...")
        
        # 헤더가 없으므로 직접 지정
        columns_url1 = ['수신년도', '수신월', '수신일', '수신시분초', '신고년도', '신고월', '신고일', '신고시분초', 'URL']
        df1 = pd.read_csv(url1_path, names=columns_url1, encoding='utf-8', low_memory=False)
        
        # hxxp -> http, hxxps -> https 변환
        df1['URL'] = df1['URL'].str.replace('hxxp://', 'http://', regex=False)
        df1['URL'] = df1['URL'].str.replace('hxxps://', 'https://', regex=False)
        df1['label'] = 1  # 악성 URL로 표시
        
        # URL과 label만 선택
        df1 = df1[['URL', 'label']]
        
        print(f"url_1.csv: {len(df1)} 개의 악성 URL 로드 완료")
        
        # URL 2 데이터 로드
        url2_path = os.path.join(self.data_dir, 'url_2.csv')
        print(f"url_2.csv 로드 중...")
        
        df2 = pd.read_csv(url2_path, encoding='utf-8', low_memory=False)
        # 날짜 열 제거하고 URL 열만 사용
        df2 = df2.rename(columns={'홈페이지주소': 'URL'})
        df2['label'] = 1  # 악성 URL로 표시
        df2 = df2[['URL', 'label']]
        
        print(f"url_2.csv: {len(df2)} 개의 악성 URL 로드 완료")
        
        # 데이터 합치기
        malicious_df = pd.concat([df1, df2], ignore_index=True)
        
        # 중복 제거
        original_count = len(malicious_df)
        malicious_df = malicious_df.drop_duplicates(subset=['URL'])
        print(f"중복 제거: {original_count} -> {len(malicious_df)} 개")
        
        # URL이 비어있거나 너무 짧은 경우 제거
        malicious_df = malicious_df[malicious_df['URL'].notna()]
        malicious_df = malicious_df[malicious_df['URL'].str.len() > 10]
        
        print(f"최종 악성 URL 수: {len(malicious_df)} 개")
        
        # 정상 URL 데이터 로드 또는 생성
        normal_urls_path = os.path.join(self.data_dir, 'normal_urls.csv')
        if os.path.exists(normal_urls_path):
            print(f"\n정상 URL 데이터 로드 중...")
            normal_df = pd.read_csv(normal_urls_path, encoding='utf-8')
            print(f"normal_urls.csv: {len(normal_df)} 개의 정상 URL 로드 완료")
        else:
            print(f"\n정상 URL 데이터가 없습니다. generate_normal_urls.py를 실행하여 생성하세요.")
            # 기본 정상 URL 사용
            normal_urls = [
                'https://www.google.com',
                'https://www.github.com',
                'https://www.microsoft.com',
                'https://www.amazon.com',
                'https://www.wikipedia.org',
                'https://www.youtube.com',
                'https://www.facebook.com',
                'https://www.twitter.com',
                'https://www.linkedin.com',
                'https://www.netflix.com'
            ]
            
            normal_df = pd.DataFrame({
                'URL': normal_urls,
                'label': [0] * len(normal_urls)  # 정상 URL로 표시
            })
        
        # 전체 데이터 합치기
        all_data = pd.concat([malicious_df, normal_df], ignore_index=True)
        
        # 데이터 섞기
        all_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\n전체 데이터셋 크기: {len(all_data)}")
        print(f"악성 URL: {len(all_data[all_data['label']==1])}")
        print(f"정상 URL: {len(all_data[all_data['label']==0])}")
        
        return all_data
    
    def prepare_training_data(self, df, test_size=0.2, random_state=42):
        """학습 데이터 준비"""
        # 학습/검증 데이터 분할
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df['label']
        )
        
        print(f"\n학습 데이터: {len(train_df)} 개")
        print(f"검증 데이터: {len(test_df)} 개")
        
        # 클래스 가중치 계산 (불균형 데이터 처리)
        y_train = train_df['label'].values
        classes = np.unique(y_train)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )
        
        class_weight_dict = dict(zip(classes, class_weights))
        print(f"\n클래스 가중치: {class_weight_dict}")
        
        return train_df, test_df, class_weight_dict
    
    def train_model(self, train_df, test_df, class_weights=None):
        """AutoGluon을 사용한 모델 학습"""
        print("\n모델 학습을 시작합니다...")
        
        # AutoGluon MultiModalPredictor 초기화
        self.predictor = MultiModalPredictor(
            label='label',
            problem_type='binary',
            eval_metric='roc_auc',
            path=self.model_path
        )
        
        # Hugging Face 모델을 사용한 학습 설정
        hyperparameters = {
            # BERT 기반 악성 URL 탐지 모델 사용
            'model.hf_text.checkpoint_name': 'bert-base-uncased',
            'optimization.learning_rate': 2e-5,
            'optimization.max_epochs': 3,
            'optimization.batch_size': 16,
            'env.per_gpu_batch_size': 16,
            'model.hf_text.text_trivial_aug_maxscale': 0.0,  # 텍스트 증강 비활성화
            'model.hf_text.max_text_len': 512,  # URL은 일반적으로 짧으므로
        }
        
        # 학습 시작
        self.predictor.fit(
            train_data=train_df,
            tuning_data=test_df,
            hyperparameters=hyperparameters,
            time_limit=600,  # 10분 제한 (실제 학습시에는 더 길게 설정)
            presets='best_quality'
        )
        
        print("모델 학습 완료!")
        
        # 모델 평가
        scores = self.predictor.evaluate(test_df, metrics=['roc_auc', 'accuracy', 'precision', 'recall', 'f1'])
        print("\n모델 성능:")
        for metric, score in scores.items():
            print(f"{metric}: {score:.4f}")
        
        return scores
    
    def save_model_info(self, scores):
        """모델 정보 저장"""
        import json
        
        model_info = {
            'model_path': self.model_path,
            'evaluation_scores': scores,
            'hyperparameters': {
                'base_model': 'bert-base-uncased',
                'problem_type': 'binary',
                'eval_metric': 'roc_auc'
            }
        }
        
        with open('model_info.json', 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        print("\n모델 정보가 model_info.json에 저장되었습니다.")
    
    def predict_urls(self, urls):
        """URL 악성 여부 예측"""
        if self.predictor is None:
            print("모델을 먼저 학습하거나 로드해주세요.")
            return None
        
        # DataFrame으로 변환
        if isinstance(urls, str):
            urls = [urls]
        
        test_data = pd.DataFrame({'URL': urls})
        
        # 예측
        predictions = self.predictor.predict(test_data)
        probabilities = self.predictor.predict_proba(test_data)
        
        results = []
        for i, url in enumerate(urls):
            results.append({
                'url': url,
                'is_malicious': bool(predictions[i]),
                'malicious_probability': float(probabilities[i][1])
            })
        
        return results
    
    def load_model(self):
        """저장된 모델 로드"""
        print(f"모델을 로드하는 중... ({self.model_path})")
        self.predictor = MultiModalPredictor.load(self.model_path)
        print("모델 로드 완료!")

def main():
    # 악성 URL 체커 초기화
    checker = MaliciousURLChecker()
    
    # 데이터 로드 및 전처리
    df = checker.load_and_preprocess_data()
    
    # 학습 데이터 준비
    train_df, test_df, class_weights = checker.prepare_training_data(df)
    
    # 모델 학습
    scores = checker.train_model(train_df, test_df, class_weights)
    
    # 모델 정보 저장
    checker.save_model_info(scores)
    
    # 예측 테스트
    print("\n=== 예측 테스트 ===")
    test_urls = [
        'https://bit.ly/suspicious-link',
        'https://www.google.com',
        'http://phishing-site.fake.com',
        'https://github.com'
    ]
    
    results = checker.predict_urls(test_urls)
    for result in results:
        status = "악성" if result['is_malicious'] else "정상"
        print(f"URL: {result['url']}")
        print(f"  상태: {status}")
        print(f"  악성 확률: {result['malicious_probability']:.2%}\n")

if __name__ == "__main__":
    main()
