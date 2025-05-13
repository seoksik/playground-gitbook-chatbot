#!/usr/bin/env python
"""
.env 파일을 생성하는 스크립트입니다.
"""

with open(".env", "w") as f:
    f.write("""# OpenAI API 키
OPENAI_API_KEY=your_openai_api_key_here

# Supabase 연결 정보
SUPABASE_URL=your_supabase_url_here
SUPABASE_ANON_KEY=your_supabase_anon_key_here

# 타겟 GitBook 정보 (선택 사항)
TARGET_GITBOOK_NAME=FETA 문서

# 웹 요청 식별자 (선택 사항)
USER_AGENT=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36
""")

print(".env 파일이 생성되었습니다. 이제 해당 파일을 편집하여 필요한 API 키와 연결 정보를 입력하세요.") 