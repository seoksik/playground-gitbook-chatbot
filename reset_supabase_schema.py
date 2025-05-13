#!/usr/bin/env python
"""
Supabase 데이터베이스 테이블과 함수를 삭제하고 재생성하는 스크립트입니다.
UUID를 사용하는 새 스키마를 적용합니다.
"""

import os
from dotenv import load_dotenv
from supabase.client import Client, create_client

# .env 파일에서 환경 변수 로드
load_dotenv()

# 환경 변수 확인
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

# 필수 환경 변수 체크
if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    missing_vars = []
    if not SUPABASE_URL:
        missing_vars.append("SUPABASE_URL")
    if not SUPABASE_ANON_KEY:
        missing_vars.append("SUPABASE_ANON_KEY")
    
    print(f"오류: 다음 환경 변수가 설정되지 않았습니다: {', '.join(missing_vars)}")
    print("create_env.py 스크립트로 생성된 .env 파일을 편집하여 필요한 값을 채워주세요.")
    exit(1)

# SQL 쿼리 정의
DROP_TABLE_QUERY = """
DROP TABLE IF EXISTS documents;
"""

DROP_FUNCTION_QUERY = """
DROP FUNCTION IF EXISTS match_documents;
"""

CREATE_TABLE_QUERY = """
CREATE TABLE IF NOT EXISTS documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  content TEXT,
  metadata JSONB,
  embedding VECTOR(1536)
);
"""

CREATE_FUNCTION_QUERY = """
CREATE OR REPLACE FUNCTION match_documents (
  query_embedding VECTOR(1536),
  match_threshold FLOAT DEFAULT 0.5,
  match_count INT DEFAULT 5
)
RETURNS TABLE (
  id UUID,
  content TEXT,
  metadata JSONB,
  similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    documents.id,
    documents.content,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) AS similarity
  FROM documents
  WHERE 1 - (documents.embedding <=> query_embedding) > match_threshold
  ORDER BY documents.embedding <=> query_embedding ASC
  LIMIT match_count;
END;
$$;
"""

def main():
    print("Supabase 데이터베이스 스키마 초기화 및 재설정을 시작합니다...")
    
    try:
        # Supabase 클라이언트 초기화
        supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        print("Supabase 연결 성공!")
        
        # 주의: Supabase REST API로는 SQL을 직접 실행할 수 없습니다.
        # 아래는 실제로 실행되지 않는 코드로, 참고용으로만 제공합니다.
        print("\n중요: 이 스크립트는 실제로 데이터베이스 작업을 수행하지 않습니다.")
        print("Supabase 대시보드의 SQL 에디터에서 다음 SQL 쿼리를 직접 실행해야 합니다:")
        
        print("\n--- 기존 테이블 및 함수 삭제 ---")
        print(DROP_FUNCTION_QUERY)
        print(DROP_TABLE_QUERY)
        
        print("\n--- 새 테이블 생성 ---")
        print(CREATE_TABLE_QUERY)
        
        print("\n--- 유사도 검색 함수 생성 ---")
        print(CREATE_FUNCTION_QUERY)
        
        print("\n--- 선택 사항: 인덱스 생성 ---")
        print("-- CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops)")
        print("-- WITH (lists = 100);")
        
        print("\n위 SQL을 Supabase의 SQL 에디터에서 실행한 후, ingest_gitbook.py 스크립트를 다시 실행하세요.")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        return

if __name__ == "__main__":
    main() 