import pandas as pd
import random

def generate_normal_urls():
    """다양한 정상 URL 생성"""
    
    # 주요 도메인들
    top_domains = [
        'google.com', 'youtube.com', 'facebook.com', 'wikipedia.org', 'amazon.com',
        'twitter.com', 'instagram.com', 'linkedin.com', 'reddit.com', 'netflix.com',
        'microsoft.com', 'apple.com', 'github.com', 'stackoverflow.com', 'medium.com',
        'naver.com', 'daum.net', 'kakao.com', 'coupang.com', 'nate.com',
        'bbc.com', 'cnn.com', 'nytimes.com', 'reuters.com', 'theguardian.com',
        'walmart.com', 'ebay.com', 'alibaba.com', 'booking.com', 'airbnb.com'
    ]
    
    # 일반적인 경로들
    common_paths = [
        '', '/home', '/about', '/contact', '/products', '/services',
        '/blog', '/news', '/help', '/support', '/privacy', '/terms',
        '/login', '/register', '/profile', '/account', '/settings',
        '/search', '/category', '/shop', '/cart', '/checkout'
    ]
    
    # 프로토콜
    protocols = ['https://', 'https://www.', 'http://', 'http://www.']
    
    normal_urls = []
    
    # 기본 도메인 URL 생성
    for domain in top_domains:
        for protocol in protocols[:2]:  # https만 사용
            normal_urls.append(f"{protocol}{domain}")
    
    # 경로가 있는 URL 생성
    for _ in range(200):
        domain = random.choice(top_domains)
        protocol = random.choice(protocols[:2])
        path = random.choice(common_paths)
        normal_urls.append(f"{protocol}{domain}{path}")
    
    # 서브도메인이 있는 URL
    subdomains = ['api', 'blog', 'shop', 'mail', 'news', 'support', 'm', 'mobile']
    for _ in range(100):
        domain = random.choice(top_domains)
        subdomain = random.choice(subdomains)
        normal_urls.append(f"https://{subdomain}.{domain}")
    
    # 쿼리 파라미터가 있는 정상 URL
    for _ in range(50):
        domain = random.choice(top_domains)
        queries = ['?page=1', '?search=product', '?lang=ko', '?ref=home', '?utm_source=google']
        query = random.choice(queries)
        normal_urls.append(f"https://www.{domain}/search{query}")
    
    # 중복 제거
    normal_urls = list(set(normal_urls))
    
    return normal_urls

def save_normal_urls():
    """정상 URL을 CSV 파일로 저장"""
    normal_urls = generate_normal_urls()
    
    df = pd.DataFrame({
        'URL': normal_urls,
        'label': [0] * len(normal_urls)
    })
    
    df.to_csv('data/normal_urls.csv', index=False, encoding='utf-8')
    print(f"정상 URL {len(normal_urls)}개를 data/normal_urls.csv에 저장했습니다.")
    
    return df

if __name__ == "__main__":
    save_normal_urls() 