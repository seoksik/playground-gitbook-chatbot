# Gitbook Q&A Chatbot

Gitbook 문서를 기반으로 질문에 답변해주는 챗봇입니다. Supabase Vector DB와 OpenAI의 임베딩 및 LLM을 활용하여 제작되었습니다.

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. 필요한 추가 패키지 설치:
```bash
pip install supabase
```

3. 환경 변수 설정:
```bash
python create_env.py
```
생성된 `.env` 파일을 편집하여 다음 내용을 입력하세요:
- `OPENAI_API_KEY`: OpenAI API 키 
- `SUPABASE_URL`: Supabase 프로젝트 URL
- `SUPABASE_ANON_KEY`: Supabase 익명 키
- `TARGET_GITBOOK_NAME`: 대상 Gitbook 이름 (선택 사항)

## Supabase 설정

1. Supabase 프로젝트 생성
2. pgvector 확장 활성화
3. 스키마 설정 방법:
   - 방법 1: Supabase SQL 에디터에서 `supabase_schema.sql` 내용을 직접 실행
   - 방법 2: 스키마 초기화 스크립트 실행 (표시된 SQL문을 Supabase SQL 에디터에 복사하여 실행)
   ```bash
   python reset_supabase_schema.py
   ```

### 타입 불일치 오류 해결

UUID 타입 오류가 발생하는 경우 다음 단계를 수행하세요:

1. `reset_supabase_schema.py`를 실행하여 필요한 SQL 명령어 확인
2. 표시된 SQL 명령어를 Supabase SQL 에디터에 복사하여 실행:
   - 기존 테이블과 함수 삭제
   - UUID 타입을 사용하는 새 테이블 생성
   - 새 매칭 함수 생성

## 사용 방법

1. Gitbook 문서 수집 및 임베딩:
```bash
python ingest_gitbook.py
```

2. 웹 인터페이스 실행:
```bash
streamlit run app.py
```

## 주요 기능

- Gitbook 문서 크롤링 및 임베딩
- 입력 질문에 대한 관련 문서 검색
- OpenAI 모델을 통한 답변 생성
- 출처 문서 표시

## 시스템 아키텍처

### 전체 시스템 구조

```mermaid
graph TB
    subgraph "External Services"
        GB[Gitbook 문서]
        OAI[OpenAI API]
        SB[Supabase Vector DB]
    end
    
    subgraph "Data Ingestion Layer"
        IG[ingest_gitbook.py]
        BS[BeautifulSoup Scraper]
        TS[Text Splitter]
        EMB[OpenAI Embeddings]
    end
    
    subgraph "Application Layer"
        APP[app.py - Streamlit UI]
        QA[Q&A Chain]
        MEM[Conversation Memory]
        RET[Vector Retriever]
    end
    
    subgraph "Storage Layer"
        VDB[(Vector Database)]
        CHAT[(Chat History)]
    end
    
    GB --> IG
    IG --> BS
    BS --> TS
    TS --> EMB
    EMB --> OAI
    EMB --> VDB
    VDB --> SB
    
    APP --> QA
    QA --> RET
    RET --> VDB
    QA --> OAI
    QA --> MEM
    MEM --> CHAT
    
    style GB fill:#e1f5fe
    style OAI fill:#fff3e0
    style SB fill:#e8f5e8
    style APP fill:#f3e5f5
```

### 데이터 흐름 아키텍처

```mermaid
flowchart LR
    subgraph "Data Sources"
        A[Gitbook Pages]
        B[Sitemap XML]
    end
    
    subgraph "Processing Pipeline"
        C[Web Scraping]
        D[Content Extraction]
        E[Text Chunking]
        F[Embedding Generation]
        G[Vector Storage]
    end
    
    subgraph "Query Processing"
        H[User Query]
        I[Query Embedding]
        J[Similarity Search]
        K[Context Retrieval]
        L[LLM Response]
    end
    
    A --> C
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    
    H --> I
    I --> J
    J --> K
    K --> L
    G --> J
    
    style A fill:#e3f2fd
    style H fill:#fff3e0
    style L fill:#e8f5e8
```

## 시퀀스 다이어그램

### 문서 수집 및 임베딩 프로세스

```mermaid
sequenceDiagram
    participant User
    participant IngestScript as ingest_gitbook.py
    participant Gitbook
    participant OpenAI
    participant Supabase
    
    User->>IngestScript: python ingest_gitbook.py 실행
    IngestScript->>Gitbook: 사이트맵 XML 요청
    Gitbook-->>IngestScript: URL 목록 반환
    
    loop 각 페이지별
        IngestScript->>Gitbook: 페이지 내용 요청
        Gitbook-->>IngestScript: HTML 내용 반환
        IngestScript->>IngestScript: BeautifulSoup으로 텍스트 추출
        IngestScript->>IngestScript: 텍스트 청킹 (1000자 단위)
    end
    
    IngestScript->>OpenAI: 임베딩 생성 요청
    OpenAI-->>IngestScript: 벡터 임베딩 반환
    IngestScript->>Supabase: 문서 + 임베딩 저장
    Supabase-->>IngestScript: 저장 완료 응답
    IngestScript-->>User: 수집 완료 메시지
```

### 사용자 질의응답 프로세스

```mermaid
sequenceDiagram
    participant User
    participant Streamlit as app.py
    participant Memory as Conversation Memory
    participant Retriever as Vector Retriever
    participant Supabase
    participant OpenAI
    
    User->>Streamlit: 질문 입력
    Streamlit->>Memory: 대화 기록 확인
    Streamlit->>OpenAI: 질문 임베딩 생성
    OpenAI-->>Streamlit: 질문 벡터 반환
    
    Streamlit->>Retriever: 유사 문서 검색 요청
    Retriever->>Supabase: match_documents() 함수 호출
    Supabase-->>Retriever: 관련 문서 청크 반환
    Retriever-->>Streamlit: 컨텍스트 문서 반환
    
    Streamlit->>OpenAI: 질문 + 컨텍스트로 답변 생성 요청
    OpenAI-->>Streamlit: 답변 텍스트 반환
    
    Streamlit->>Memory: 질문-답변 쌍 저장
    Streamlit->>Streamlit: 추천 질문 생성
    Streamlit-->>User: 답변 + 출처 + 추천 질문 표시
```

### 대화 히스토리 관리 프로세스

```mermaid
sequenceDiagram
    participant User
    participant Streamlit as app.py
    participant FileSystem as chat_history.json
    participant Memory as Session Memory
    
    User->>Streamlit: 앱 시작
    Streamlit->>FileSystem: 기존 대화 히스토리 로드
    FileSystem-->>Streamlit: 저장된 대화 목록 반환
    
    User->>Streamlit: 질문-답변 진행
    Streamlit->>Memory: 세션 메모리에 저장
    
    alt 새 대화 시작
        User->>Streamlit: "새 대화 시작" 클릭
        Streamlit->>FileSystem: 현재 대화 저장
        Streamlit->>Memory: 메모리 초기화
        Streamlit-->>User: 새 대화 화면 표시
    else 기존 대화 선택
        User->>Streamlit: 저장된 대화 선택
        Streamlit->>FileSystem: 선택된 대화 로드
        Streamlit->>Memory: 메모리에 대화 복원
        Streamlit-->>User: 선택된 대화 화면 표시
    end
```

## 파일 구조

- `app.py`: Streamlit 웹 인터페이스
- `ingest_gitbook.py`: 문서 수집 및 임베딩 스크립트
- `supabase_schema.sql`: Supabase 데이터베이스 스키마
- `reset_supabase_schema.py`: Supabase 스키마 초기화 스크립트
- `requirements.txt`: 필요 패키지 목록
- `create_env.py`: 환경 변수 파일 생성 도우미

## 문제 해결

- **GitbookLoader가 작동하지 않는 경우**: 스크립트는 BeautifulSoup 기반 추출로 자동 대체됩니다.
- **Supabase 연결 오류**: `.env` 파일 설정 및 Supabase 프로젝트 설정 확인
- **임베딩 오류**: OpenAI API 키 유효성 확인
- **UUID 타입 오류**: Supabase 테이블 구조가 최신 스키마와 일치하지 않을 때 발생합니다. `reset_supabase_schema.py`를 참고하여 스키마를 업데이트하세요. 