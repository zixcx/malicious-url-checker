import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from autogluon.multimodal import MultiModalPredictor
import warnings
import torch
import gc

warnings.filterwarnings('ignore')

class AdvancedMaliciousURLTrainer:
    def __init__(self, data_dir='data', model_path='./malicious_url_model_gpu'):
        self.data_dir = data_dir
        self.model_path = model_path
        self.predictor = None
        
        # GPU 확인
        if torch.cuda.is_available():
            print(f"GPU 사용 가능: {torch.cuda.get_device_name(0)}")
            print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("GPU를 사용할 수 없습니다. CPU로 학습합니다.")
    
    def load_all_data(self):
        """모든 데이터 로드 (대용량 처리)"""
        print("데이터를 로드하고 있습니다...")
        
        all_dfs = []
        
        # URL 1 데이터 로드 (청크 단위로 처리)
        url1_path = os.path.join(self.data_dir, 'url_1.csv')
        print(f"url_1.csv 로드 중 (청크 단위)...")
        
        columns_url1 = ['수신년도', '수신월', '수신일', '수신시분초', '신고년도', '신고월', '신고일', '신고시분초', 'URL']
        
        # 청크 단위로 읽어서 메모리 효율적으로 처리
        chunk_size = 50000
        chunks = []
        
        for chunk in pd.read_csv(url1_path, names=columns_url1, encoding='utf-8', 
                                 low_memory=False, chunksize=chunk_size):
            # URL 정규화
            chunk['URL'] = chunk['URL'].str.replace('hxxp://', 'http://', regex=False)
            chunk['URL'] = chunk['URL'].str.replace('hxxps://', 'https://', regex=False)
            chunk['label'] = 1
            chunk = chunk[['URL', 'label']]
            
            # 유효한 URL만 선택
            chunk = chunk[chunk['URL'].notna()]
            chunk = chunk[chunk['URL'].str.len() > 10]
            
            chunks.append(chunk)
        
        df1 = pd.concat(chunks, ignore_index=True)
        all_dfs.append(df1)
        print(f"url_1.csv: {len(df1)} 개의 악성 URL 로드 완료")
        
        # 메모리 정리
        del chunks
        gc.collect()
        
        # URL 2 데이터 로드
        url2_path = os.path.join(self.data_dir, 'url_2.csv')
        print(f"url_2.csv 로드 중...")
        
        df2 = pd.read_csv(url2_path, encoding='utf-8', low_memory=False)
        df2 = df2.rename(columns={'홈페이지주소': 'URL'})
        df2['label'] = 1
        df2 = df2[['URL', 'label']]
        df2 = df2[df2['URL'].notna()]
        df2 = df2[df2['URL'].str.len() > 10]
        
        all_dfs.append(df2)
        print(f"url_2.csv: {len(df2)} 개의 악성 URL 로드 완료")
        
        # 정상 URL 데이터 로드
        normal_urls_path = os.path.join(self.data_dir, 'normal_urls.csv')
        if os.path.exists(normal_urls_path):
            print(f"정상 URL 데이터 로드 중...")
            normal_df = pd.read_csv(normal_urls_path, encoding='utf-8')
            all_dfs.append(normal_df)
            print(f"normal_urls.csv: {len(normal_df)} 개의 정상 URL 로드 완료")
        
        # 모든 데이터 합치기
        all_data = pd.concat(all_dfs, ignore_index=True)
        
        # 중복 제거
        original_count = len(all_data)
        all_data = all_data.drop_duplicates(subset=['URL'])
        print(f"\n중복 제거: {original_count} -> {len(all_data)} 개")
        
        # 데이터 섞기
        all_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\n전체 데이터셋 크기: {len(all_data)}")
        print(f"악성 URL: {len(all_data[all_data['label']==1])}")
        print(f"정상 URL: {len(all_data[all_data['label']==0])}")
        
        return all_data
    
    def advanced_train(self, df):
        """고급 학습 설정"""
        # 학습/검증/테스트 데이터 분할
        train_val_df, test_df = train_test_split(
            df, test_size=0.1, random_state=42, stratify=df['label']
        )
        train_df, val_df = train_test_split(
            train_val_df, test_size=0.11, random_state=42, stratify=train_val_df['label']
        )
        
        print(f"\n학습 데이터: {len(train_df)} 개")
        print(f"검증 데이터: {len(val_df)} 개")
        print(f"테스트 데이터: {len(test_df)} 개")
        
        # 클래스 가중치 계산
        y_train = train_df['label'].values
        classes = np.unique(y_train)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )
        
        class_weight_dict = dict(zip(classes, class_weights))
        print(f"\n클래스 가중치: {class_weight_dict}")
        
        # AutoGluon MultiModalPredictor 초기화
        self.predictor = MultiModalPredictor(
            label='label',
            problem_type='binary',
            eval_metric='roc_auc',
            path=self.model_path
        )
        
        # 고급 하이퍼파라미터 설정
        hyperparameters = {
            # 더 강력한 BERT 모델 사용 (보안 특화 모델이 있다면 사용)
            'model.hf_text.checkpoint_name': 'microsoft/deberta-v3-base',  # 더 나은 성능
            'optimization.learning_rate': 1e-5,
            'optimization.weight_decay': 0.01,
            'optimization.max_epochs': 10,  # 더 많은 에포크
            'optimization.batch_size': 32,  # GPU 메모리가 충분하면 증가
            'optimization.warmup_steps': 0.1,
            'optimization.gradient_clip_val': 1.0,
            'env.per_gpu_batch_size': 32,
            'env.num_gpus': 1,  # GPU 수
            'model.hf_text.pooling_mode': 'cls',
            'model.hf_text.text_trivial_aug_maxscale': 0.1,  # 약간의 텍스트 증강
            'model.hf_text.max_text_len': 256,  # URL은 대부분 짧음
            'model.hf_text.gradient_checkpointing': True,  # 메모리 효율성
            'optimization.top_k': 3,  # 상위 3개 체크포인트 저장
            'optimization.save_best': True,
            'optimization.patience': 5,
        }
        
        # URL 특화 feature engineering을 위한 전처리 추가
        def preprocess_url(url):
            """URL 전처리 및 특징 추출"""
            # 도메인, 경로, 쿼리 등 분리
            # 특수문자 비율, URL 길이 등 추가 특징
            return url
        
        # 학습 시작
        print("\n고성능 GPU 학습을 시작합니다...")
        self.predictor.fit(
            train_data=train_df,
            tuning_data=val_df,
            hyperparameters=hyperparameters,
            time_limit=7200,  # 2시간
            presets='best_quality',
            verbosity=3  # 상세 로그
        )
        
        print("모델 학습 완료!")
        
        # 상세 평가
        print("\n=== 검증 데이터 평가 ===")
        val_scores = self.predictor.evaluate(val_df, metrics=['roc_auc', 'accuracy', 'precision', 'recall', 'f1'])
        for metric, score in val_scores.items():
            print(f"{metric}: {score:.4f}")
        
        print("\n=== 테스트 데이터 평가 ===")
        test_scores = self.predictor.evaluate(test_df, metrics=['roc_auc', 'accuracy', 'precision', 'recall', 'f1'])
        for metric, score in test_scores.items():
            print(f"{metric}: {score:.4f}")
        
        # Confusion Matrix
        self._print_confusion_matrix(test_df)
        
        return test_scores
    
    def _print_confusion_matrix(self, test_df):
        """혼동 행렬 출력"""
        from sklearn.metrics import confusion_matrix, classification_report
        
        y_true = test_df['label'].values
        y_pred = self.predictor.predict(test_df)
        
        cm = confusion_matrix(y_true, y_pred)
        print("\n=== 혼동 행렬 ===")
        print("[[TN  FP]")
        print(" [FN  TP]]")
        print(cm)
        
        print("\n=== 상세 분류 리포트 ===")
        print(classification_report(y_true, y_pred, 
                                   target_names=['정상 URL', '악성 URL']))
    
    def analyze_errors(self, test_df, num_samples=10):
        """오분류 분석"""
        print(f"\n=== 오분류 분석 (상위 {num_samples}개) ===")
        
        y_true = test_df['label'].values
        y_pred = self.predictor.predict(test_df)
        proba = self.predictor.predict_proba(test_df)
        
        # 오분류된 샘플 찾기
        errors = test_df[y_true != y_pred].copy()
        errors['true_label'] = y_true[y_true != y_pred]
        errors['pred_label'] = y_pred[y_true != y_pred]
        errors['confidence'] = proba[y_true != y_pred].max(axis=1)
        
        # False Positive (정상을 악성으로)
        fp_samples = errors[errors['true_label'] == 0].nlargest(num_samples, 'confidence')
        if len(fp_samples) > 0:
            print("\nFalse Positive (정상 → 악성):")
            for idx, row in fp_samples.iterrows():
                print(f"URL: {row['URL'][:100]}...")
                print(f"신뢰도: {row['confidence']:.3f}\n")
        
        # False Negative (악성을 정상으로)
        fn_samples = errors[errors['true_label'] == 1].nlargest(num_samples, 'confidence')
        if len(fn_samples) > 0:
            print("\nFalse Negative (악성 → 정상):")
            for idx, row in fn_samples.iterrows():
                print(f"URL: {row['URL'][:100]}...")
                print(f"신뢰도: {row['confidence']:.3f}\n")

def main():
    # 트레이너 초기화
    trainer = AdvancedMaliciousURLTrainer()
    
    # 데이터 로드
    df = trainer.load_all_data()
    
    # 고급 학습
    scores = trainer.advanced_train(df)
    
    # 오류 분석
    # trainer.analyze_errors(test_df)
    
    print("\n학습이 완료되었습니다!")
    print(f"모델이 {trainer.model_path}에 저장되었습니다.")

if __name__ == "__main__":
    main() 