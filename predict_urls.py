import pandas as pd
import numpy as np
from autogluon.multimodal import MultiModalPredictor
import argparse
import os
import json
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')

class URLPredictor:
    def __init__(self, model_path='./malicious_url_model'):
        """URL 예측기 초기화"""
        self.model_path = model_path
        self.predictor = None
        self.load_model()
    
    def load_model(self):
        """모델 로드"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델을 찾을 수 없습니다: {self.model_path}")
        
        print(f"모델을 로드하는 중... ({self.model_path})")
        self.predictor = MultiModalPredictor.load(self.model_path)
        print("모델 로드 완료!")
    
    def analyze_url_features(self, url):
        """URL의 특징 분석"""
        features = {}
        
        try:
            parsed = urlparse(url)
            
            # 기본 특징
            features['도메인'] = parsed.netloc
            features['경로'] = parsed.path if parsed.path else '/'
            features['쿼리'] = '있음' if parsed.query else '없음'
            features['프로토콜'] = parsed.scheme
            
            # 의심스러운 특징
            suspicious_features = []
            
            # URL 길이
            if len(url) > 100:
                suspicious_features.append("매우 긴 URL")
            
            # 특수문자 비율
            special_chars = sum(1 for c in url if not c.isalnum() and c not in ':/.-_?=&')
            if special_chars / len(url) > 0.2:
                suspicious_features.append("특수문자 과다")
            
            # IP 주소 사용
            if any(c.isdigit() for c in parsed.netloc.split('.')) and parsed.netloc.count('.') == 3:
                try:
                    # IP 주소 형식 확인
                    parts = parsed.netloc.split('.')
                    if all(0 <= int(part) <= 255 for part in parts if part.isdigit()):
                        suspicious_features.append("IP 주소 사용")
                except:
                    pass
            
            # 단축 URL 서비스
            shorteners = ['bit.ly', 'tinyurl.com', 'goo.gl', 'ow.ly', 't.co', 'is.gd', 'buff.ly']
            if any(shortener in parsed.netloc for shortener in shorteners):
                suspicious_features.append("URL 단축 서비스 사용")
            
            # 서브도메인 개수
            subdomain_count = parsed.netloc.count('.') - 1
            if subdomain_count > 2:
                suspicious_features.append(f"많은 서브도메인 ({subdomain_count}개)")
            
            # @ 기호 (피싱에서 자주 사용)
            if '@' in url:
                suspicious_features.append("@ 기호 포함")
            
            # 의심스러운 TLD
            suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.click', '.download', '.review']
            if any(url.endswith(tld) for tld in suspicious_tlds):
                suspicious_features.append("의심스러운 TLD")
            
            features['의심스러운 특징'] = suspicious_features if suspicious_features else ['없음']
            
        except Exception as e:
            features['분석 오류'] = str(e)
        
        return features
    
    def predict_single_url(self, url, show_analysis=True):
        """단일 URL 예측"""
        # 예측
        df = pd.DataFrame({'URL': [url]})
        prediction = self.predictor.predict(df)[0]
        probability = self.predictor.predict_proba(df)[0]
        
        # 결과
        is_malicious = bool(prediction)
        malicious_prob = float(probability[1])
        
        result = {
            'url': url,
            'is_malicious': is_malicious,
            'malicious_probability': malicious_prob,
            'verdict': '악성' if is_malicious else '정상',
            'confidence': 'HIGH' if abs(malicious_prob - 0.5) > 0.3 else 'MEDIUM' if abs(malicious_prob - 0.5) > 0.1 else 'LOW'
        }
        
        if show_analysis:
            result['features'] = self.analyze_url_features(url)
        
        return result
    
    def predict_batch(self, urls, output_file=None):
        """여러 URL 일괄 예측"""
        results = []
        
        print(f"\n{len(urls)}개의 URL을 분석 중...")
        
        # 일괄 예측
        df = pd.DataFrame({'URL': urls})
        predictions = self.predictor.predict(df)
        probabilities = self.predictor.predict_proba(df)
        
        for i, url in enumerate(urls):
            result = {
                'url': url,
                'is_malicious': bool(predictions[i]),
                'malicious_probability': float(probabilities[i][1]),
                'verdict': '악성' if predictions[i] else '정상'
            }
            results.append(result)
        
        # 파일로 저장
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"결과가 {output_file}에 저장되었습니다.")
        
        return results
    
    def predict_from_file(self, input_file, output_file=None):
        """파일에서 URL 읽어서 예측"""
        # 파일 읽기
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file, encoding='utf-8')
            if 'URL' in df.columns:
                urls = df['URL'].tolist()
            elif 'url' in df.columns:
                urls = df['url'].tolist()
            else:
                # 첫 번째 열을 URL로 가정
                urls = df.iloc[:, 0].tolist()
        else:
            # 텍스트 파일로 가정
            with open(input_file, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip()]
        
        print(f"{input_file}에서 {len(urls)}개의 URL을 로드했습니다.")
        
        # 예측
        return self.predict_batch(urls, output_file)

def print_result(result):
    """결과를 보기 좋게 출력"""
    print("\n" + "="*60)
    print(f"URL: {result['url']}")
    print(f"판정: {result['verdict']}")
    print(f"악성 확률: {result['malicious_probability']:.1%}")
    print(f"신뢰도: {result.get('confidence', 'N/A')}")
    
    if 'features' in result:
        print("\n[URL 특징 분석]")
        for key, value in result['features'].items():
            if key == '의심스러운 특징' and isinstance(value, list):
                print(f"  {key}: {', '.join(value)}")
            else:
                print(f"  {key}: {value}")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='URL 악성 여부 예측')
    parser.add_argument('--url', type=str, help='예측할 단일 URL')
    parser.add_argument('--file', type=str, help='URL이 포함된 입력 파일')
    parser.add_argument('--output', type=str, help='결과를 저장할 파일')
    parser.add_argument('--model', type=str, default='./malicious_url_model', help='모델 경로')
    parser.add_argument('--no-analysis', action='store_true', help='URL 특징 분석 생략')
    
    args = parser.parse_args()
    
    # 예측기 초기화
    predictor = URLPredictor(model_path=args.model)
    
    if args.url:
        # 단일 URL 예측
        result = predictor.predict_single_url(args.url, show_analysis=not args.no_analysis)
        print_result(result)
        
    elif args.file:
        # 파일에서 일괄 예측
        results = predictor.predict_from_file(args.file, args.output)
        
        # 요약 통계
        malicious_count = sum(1 for r in results if r['is_malicious'])
        print(f"\n[예측 요약]")
        print(f"전체 URL: {len(results)}개")
        print(f"악성 URL: {malicious_count}개 ({malicious_count/len(results)*100:.1f}%)")
        print(f"정상 URL: {len(results)-malicious_count}개 ({(len(results)-malicious_count)/len(results)*100:.1f}%)")
        
        # 상위 5개 악성 URL 출력
        malicious_urls = sorted([r for r in results if r['is_malicious']], 
                               key=lambda x: x['malicious_probability'], reverse=True)[:5]
        
        if malicious_urls:
            print(f"\n[가장 위험한 URL Top 5]")
            for i, result in enumerate(malicious_urls, 1):
                print(f"{i}. {result['url'][:80]}... (악성 확률: {result['malicious_probability']:.1%})")
        
    else:
        # 대화형 모드
        print("URL 악성 여부 예측 프로그램")
        print("'quit' 또는 'exit'를 입력하면 종료합니다.\n")
        
        while True:
            url = input("URL을 입력하세요: ").strip()
            
            if url.lower() in ['quit', 'exit']:
                print("프로그램을 종료합니다.")
                break
            
            if not url:
                continue
            
            try:
                result = predictor.predict_single_url(url, show_analysis=not args.no_analysis)
                print_result(result)
            except Exception as e:
                print(f"오류 발생: {e}")

if __name__ == "__main__":
    main() 