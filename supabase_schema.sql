-- 0. pgvector 확장 활성화 (Supabase Dashboard -> Database -> Extensions 에서 활성화)
-- CREATE EXTENSION IF NOT EXISTS vector;

-- 1. documents 테이블 생성
CREATE TABLE IF NOT EXISTS documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  content TEXT, -- 문서 내용
  metadata JSONB, -- 추가 정보 (예: 출처 URL, 제목 등)
  embedding VECTOR(1536) -- OpenAI text-embedding-ada-002 모델의 차원 수
);

-- 2. 유사도 검색 함수 생성 (매개변수 순서 수정)
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
    1 - (documents.embedding <=> query_embedding) AS similarity -- 코사인 유사도 (1 - 코사인 거리)
  FROM documents
  WHERE 1 - (documents.embedding <=> query_embedding) > match_threshold
  ORDER BY documents.embedding <=> query_embedding ASC -- 거리가 짧을수록 유사하므로 ASC
  LIMIT match_count;
END;
$$;

-- 인덱스 생성 (선택 사항이지만 대량 데이터 검색 성능 향상에 도움)
-- CREATE INDEX IF NOT EXISTS ON documents USING ivfflat (embedding vector_cosine_ops)
-- WITH (lists = 100); -- lists 값은 데이터 크기에 따라 조정

-- 또는 HNSW 인덱스 (더 정확하고 빠를 수 있지만 빌드 시간이 김)
-- CREATE INDEX IF NOT EXISTS ON documents USING hnsw (embedding vector_cosine_ops);