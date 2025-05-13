import os
import requests
import xmltodict # 사이트맵 파싱용
from dotenv import load_dotenv
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from time import sleep

from langchain_community.document_loaders import GitbookLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.supabase import SupabaseVectorStore
from langchain_core.documents import Document
from supabase.client import Client, create_client

load_dotenv()

# 환경 변수 확인
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

# 필수 환경 변수 체크
if not OPENAI_API_KEY or not SUPABASE_URL or not SUPABASE_ANON_KEY:
    missing_vars = []
    if not OPENAI_API_KEY:
        missing_vars.append("OPENAI_API_KEY")
    if not SUPABASE_URL:
        missing_vars.append("SUPABASE_URL")
    if not SUPABASE_ANON_KEY:
        missing_vars.append("SUPABASE_ANON_KEY")
    
    print(f"오류: 다음 환경 변수가 설정되지 않았습니다: {', '.join(missing_vars)}")
    print("create_env.py 스크립트로 생성된 .env 파일을 편집하여 필요한 값을 채워주세요.")
    exit(1)

# Supabase 클라이언트 초기화
supabase: Client = None
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    # 연결 테스트
    test_query = supabase.table("documents").select("id").limit(1).execute()
    print("Supabase 연결 성공!")
except Exception as e:
    print(f"Supabase 클라이언트 초기화 또는 연결 테스트 실패: {e}")
    print("다음 사항을 확인해주세요:")
    print("1. .env 파일에 올바른 SUPABASE_URL 및 SUPABASE_ANON_KEY가 설정되어 있는지")
    print("2. Supabase 프로젝트가 활성화되어 있는지")
    print("3. 'documents' 테이블이 생성되어 있고, 접근 권한이 있는지")
    print("4. pgvector 확장이 활성화되어 있는지")
    print("\nSupabase 스키마 설정은 supabase_schema.sql 파일을 참고하세요.")
    exit(1)

def get_urls_from_sitemap(sitemap_url: str) -> List[str]:
    """사이트맵 XML에서 모든 <loc> URL을 추출합니다 (xmltodict 사용)."""
    urls = []
    try:
        response = requests.get(sitemap_url, timeout=15)
        response.raise_for_status()
        sitemap_dict = xmltodict.parse(response.content)
        
        # sitemap 구조에 따라 경로 조정 필요할 수 있음
        # 예: sitemap_dict['urlset']['url']
        url_entries = sitemap_dict.get('urlset', {}).get('url', [])
        if not isinstance(url_entries, list): # 단일 url인 경우 리스트로 변환
            url_entries = [url_entries]

        for entry in url_entries:
            if isinstance(entry, dict) and 'loc' in entry:
                urls.append(entry['loc'])
        
        print(f"Extracted {len(urls)} URLs from {sitemap_url}")
    except requests.RequestException as e:
        print(f"Error fetching sitemap {sitemap_url}: {e}")
    except xmltodict.expat.ExpatError as e: # xmltodict 파싱 에러
        print(f"Error parsing sitemap XML from {sitemap_url} with xmltodict: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while processing sitemap {sitemap_url}: {e}")
    return urls

def extract_content_with_bs4(url: str, content_selector: str = "article.page-body") -> Document:
    """
    BeautifulSoup을 사용하여 웹 페이지 내용을 추출합니다.
    
    Args:
        url: 내용을 추출할 웹 페이지 URL
        content_selector: 내용을 추출할 HTML 요소의 CSS 셀렉터
        
    Returns:
        내용이 추출된 Document 객체 또는 None (내용 추출 실패시)
    """
    try:
        headers = {
            "User-Agent": os.getenv("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "lxml")
        
        # 페이지 제목 추출
        title_tag = soup.find("title")
        title = title_tag.get_text() if title_tag else "제목 없음"
        
        # 본문 내용 추출을 시도할 셀렉터 목록
        selectors_to_try = [
            content_selector,         # 사용자 지정 셀렉터
            "article",                # 일반적인 본문 요소
            "main",                   # 메인 콘텐츠 영역
            "div.content",            # 일반적인 내용 컨테이너
            "div.markdown",           # GitBook 마크다운 영역
            "div[role='main']",       # 메인 역할을 하는 div
            "body"                    # 최후의 수단으로 전체 본문
        ]
        
        content = ""
        for selector in selectors_to_try:
            content_element = soup.select_one(selector)
            if content_element and content_element.get_text(strip=True):
                # 불필요한 요소 제거 (선택 사항, 사이트에 따라 조정 필요)
                for unwanted in content_element.select("nav, footer, script, style, aside, .sidebar, .navigation"):
                    unwanted.decompose()
                    
                content = content_element.get_text(separator="\n", strip=True)
                print(f"Content extracted using selector: {selector}")
                break
        
        if not content:
            print(f"No content found in {url} using any CSS selectors")
            return None
        
        # 메타데이터와 함께 Document 객체 생성
        metadata = {
            "source": url,
            "title": title,
            "selector_used": selector if 'selector' in locals() else None
        }
        
        return Document(page_content=content, metadata=metadata)
    
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return None

def ingest_documents(
    gitbook_base_url: str,
    sitemap_xml_url: str = None,
    use_sitemap_only: bool = True,
    content_selector: str = "article.page-body", # docs.fe-ta.com 에 맞춘 selector
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    clear_existing_data: bool = False,
    use_bs4_extractor: bool = True,  # BeautifulSoup 사용 여부 플래그 추가
    request_delay: float = 0.5  # 요청 간 딜레이 (초)
) -> None:
    """
    Gitbook 문서를 로드하고 Supabase에 임베딩하여 저장합니다.
    """
    print(f"Starting ingestion for Gitbook: {gitbook_base_url}")
    all_langchain_docs: List[Document] = []

    if clear_existing_data:
        print("Clearing existing documents from Supabase table 'documents'...")
        try:
            delete_response = supabase.table("documents").delete().neq("id", -1).execute() # 모든 행 삭제
            print(f"Deletion response: {delete_response}")
            print("Existing documents cleared.")
        except Exception as e:
            print(f"Error clearing existing documents: {e}")


    page_urls_to_load = []
    if sitemap_xml_url:
        print(f"Attempting to load document URLs from sitemap: {sitemap_xml_url}")
        page_urls_to_load = get_urls_from_sitemap(sitemap_xml_url)
        if not page_urls_to_load:
            print("No URLs found in sitemap or sitemap could not be processed.")
            if use_sitemap_only:
                print("Exiting as use_sitemap_only is True and no URLs found in sitemap.")
                return
    
    if not page_urls_to_load and not use_sitemap_only:
        # GitbookLoader의 자체 크롤링은 특정 Gitbook 구현에 따라 불안정할 수 있으므로
        # 사이트맵 사용을 권장합니다. 이 부분은 예비용입니다.
        print(f"Sitemap not used or yielded no URLs. Attempting to use GitbookLoader with load_all_paths=True (may be slow/unreliable).")
        print(f"Base URL for GitbookLoader: {gitbook_base_url}")
        try:
            # 주의: load_all_paths=True는 매우 많은 요청을 발생시킬 수 있고, 대상 사이트 구조에 따라 실패할 수 있음
            loader = GitbookLoader(
                web_page=gitbook_base_url, # GitBook의 기본 URL
                load_all_paths=True,
                content_selector=content_selector,
                # base_url=gitbook_base_url, # load_all_paths 사용 시 상대경로 해석에 도움
                # requests_per_second=0.5 # 요청 속도 제한 (초당 요청 수)
            )
            # temp_docs = loader.load() # 시간이 매우 오래 걸릴 수 있음
            # print(f"GitbookLoader with load_all_paths found {len(temp_docs)} potential documents.")
            # all_langchain_docs.extend(temp_docs)
            print("GitbookLoader with load_all_paths=True is disabled by default in this script due to potential issues. Use sitemap or provide specific URLs.")
            # 만약 위 loader.load()를 활성화한다면, 아래 로직은 page_urls_to_load가 비어있을 때만 실행되도록 조정 필요
        except Exception as e:
            print(f"Error using GitbookLoader with load_all_paths=True from {gitbook_base_url}: {e}")
    
    if page_urls_to_load: # 사이트맵에서 가져온 URL이 있다면, 그것들을 우선적으로 로드
        print(f"Loading content from {len(page_urls_to_load)} URLs found in sitemap...")
        for i, page_url in enumerate(page_urls_to_load):
            print(f"Processing URL ({i+1}/{len(page_urls_to_load)}): {page_url}")
            
            # 요청 간 딜레이 추가 (서버 부하 방지)
            if i > 0:
                sleep(request_delay)
                
            if use_bs4_extractor:
                # BeautifulSoup을 사용하여 내용 추출
                doc = extract_content_with_bs4(page_url, content_selector)
                if doc:
                    all_langchain_docs.append(doc)
                    print(f"Successfully loaded content from {page_url} using BeautifulSoup")
                else:
                    print(f"Failed to extract content from {page_url} using BeautifulSoup")
            else:
                # 기존 GitbookLoader 사용
                try:
                    loader = GitbookLoader(
                        web_page=page_url,
                        load_all_paths=False, # 개별 URL 로드
                        content_selector=content_selector,
                        # requests_kwargs={'timeout': 20} # 타임아웃 설정
                    )
                    docs_from_page = loader.load()
                    if docs_from_page:
                        # GitbookLoader는 각 페이지를 단일 Document로 반환하는 경향이 있음
                        # 메타데이터에 URL 등을 잘 넣어주는지 확인 필요
                        for doc in docs_from_page: # loader.load()는 리스트를 반환
                            if not doc.metadata.get("source"): # source가 없다면 채워줌
                                doc.metadata["source"] = page_url
                        all_langchain_docs.extend(docs_from_page)
                        print(f"Successfully loaded content from {page_url} using GitbookLoader. Documents added: {len(docs_from_page)}")
                    else:
                        print(f"No content loaded from {page_url} using GitbookLoader.")
                except Exception as e:
                    print(f"Error loading content from {page_url} using GitbookLoader: {e}")
                    continue
    
    if not all_langchain_docs:
        print("No documents were loaded. Exiting.")
        return

    print(f"Total documents loaded before splitting: {len(all_langchain_docs)}")

    # 문서 내용이 너무 짧은 경우 필터링 (선택 사항)
    min_doc_length = 30 # 최소 문서 길이 (글자 수)
    filtered_docs = [doc for doc in all_langchain_docs if len(doc.page_content.strip()) >= min_doc_length]
    if len(filtered_docs) < len(all_langchain_docs):
        print(f"Filtered out {len(all_langchain_docs) - len(filtered_docs)} short or empty documents.")
    
    if not filtered_docs:
        print("No documents remaining after filtering. Exiting.")
        return

    # 2. 문서 분할
    print(f"Splitting {len(filtered_docs)} documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    documents_chunks = text_splitter.split_documents(filtered_docs)
    print(f"Split into {len(documents_chunks)} chunks.")

    if not documents_chunks:
        print("No chunks to process. Exiting.")
        return

    # 3. 임베딩 모델 초기화
    print("Initializing OpenAI embeddings...")
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    except Exception as e:
        print(f"Error initializing OpenAI embeddings: {e}")
        return

    # 4. Supabase Vector Store에 저장
    print(f"Storing {len(documents_chunks)} chunks/embeddings in Supabase...")
    try:
        # 테이블이 이미 존재하는지 확인하고, 테이블 구조가 일치하지 않으면 테이블을 재생성
        try:
            # 테이블 구조 확인 전에 테이블 존재 여부 먼저 확인
            table_check = supabase.table("documents").select("id").limit(1).execute()
            print("Existing 'documents' table found.")
        except Exception as table_err:
            print(f"Error checking documents table: {table_err}")
            print("Creating new documents table with UUID primary key...")
            
            # 테이블이 없거나 접근할 수 없는 경우 새 테이블 생성 시도
            try:
                # SQL 실행을 통해 테이블 생성 (REST API로는 제한적임)
                # 이 부분은 실제 Supabase에서는 SQL 편집기로 직접 실행하는 것이 좋습니다.
                print("Please create the table manually in Supabase SQL editor using supabase_schema.sql file.")
                print("Attempting to continue with existing table structure...")
            except Exception as create_err:
                print(f"Error creating table: {create_err}")
        
        # SupabaseVectorStore로 문서 저장
        vector_store = SupabaseVectorStore.from_documents(
            documents=documents_chunks,
            embedding=embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents", # 이 함수는 검색 시 사용됨, 저장 시에는 직접 사용되지 않음
            # 주의: Supabase 테이블 스키마가 변경되면 이 부분도 업데이트 필요
        )
        print("Ingestion complete! All chunks stored in Supabase.")
    except Exception as e:
        print(f"Error during Supabase ingestion: {e}")
        print("\n가능한 원인:")
        print("1. 'documents' 테이블이 존재하지 않거나 schema가 일치하지 않음")
        print("2. pgvector 확장이 활성화되지 않음")
        print("3. id 필드 타입 불일치 (UUID vs BIGINT)")
        print("4. match_documents 함수가 없거나 정의가 일치하지 않음")
        print("\n해결 방법:")
        print("1. Supabase 대시보드의 SQL 편집기에서 supabase_schema.sql 파일 내용을 실행하세요.")
        print("2. pgvector 확장이 활성화되어 있는지 확인하세요.")
        print("3. 테이블 권한 설정을 확인하세요.")
        print("4. 기존 테이블을 삭제하고 새로 생성하는 것이 가장 확실한 해결책입니다.")

if __name__ == "__main__":
    TARGET_GITBOOK_BASE_URL = "https://docs.fe-ta.com/"
    SITEMAP_XML_URL = "https://docs.fe-ta.com/sitemap-pages.xml"
    
    # docs.fe-ta.com 페이지의 주요 콘텐츠는 <article class="page-body"> 안에 있는 것으로 보입니다.
    # GitbookLoader는 기본적으로 'main' 태그를 찾으려 할 수 있으므로, 명시적으로 지정하는 것이 좋습니다.
    CONTENT_SELECTOR_FOR_FETA = "article.page-body" 
    # 또는 더 구체적으로: "main div[role='main'] article.page-body" 등 실제 구조에 맞게
    # 간단히 "article"도 시도해볼 수 있습니다.

    # True로 설정하면, 스크립트 실행 시 Supabase의 'documents' 테이블 내용이 모두 삭제된 후 새로 추가됩니다.
    # False로 설정하면, 기존 데이터는 유지되고 새로운 데이터가 추가됩니다 (중복 가능성 있음).
    # 첫 실행 시 또는 전체 갱신 시 True, 부분 추가 시 False (단, 중복 처리 로직은 현재 없음)
    CLEAR_EXISTING_DATA_ON_INGEST = True
    
    # BeautifulSoup 추출 기능 사용 여부 (GitbookLoader가 작동하지 않을 때 True로 설정)
    USE_BS4_EXTRACTOR = True
    
    # 웹 요청 간 딜레이 (초) - 서버 부하 방지를 위해
    REQUEST_DELAY = 1.0

    print(f"Target GitBook URL: {TARGET_GITBOOK_BASE_URL}")
    print(f"Sitemap URL: {SITEMAP_XML_URL}")
    print(f"Content Selector: {CONTENT_SELECTOR_FOR_FETA}")
    print(f"Clear existing data: {CLEAR_EXISTING_DATA_ON_INGEST}")
    print(f"Using BeautifulSoup extractor: {USE_BS4_EXTRACTOR}")
    print(f"Request delay: {REQUEST_DELAY} seconds")

    user_confirm = input("Proceed with ingestion? (yes/no): ")
    if user_confirm.lower() == 'yes':
        ingest_documents(
            gitbook_base_url=TARGET_GITBOOK_BASE_URL,
            sitemap_xml_url=SITEMAP_XML_URL,
            use_sitemap_only=True, # 사이트맵이 정확하다면 True 권장
            content_selector=CONTENT_SELECTOR_FOR_FETA,
            clear_existing_data=CLEAR_EXISTING_DATA_ON_INGEST,
            use_bs4_extractor=USE_BS4_EXTRACTOR,
            request_delay=REQUEST_DELAY
        )
    else:
        print("Ingestion cancelled by user.")