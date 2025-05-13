import streamlit as st

# 스트림릿 페이지 구성 설정 (반드시 다른 st 명령 전에 호출해야 함)
st.set_page_config(page_title="Gitbook Q&A Chatbot", layout="wide", initial_sidebar_state="expanded")

import os
import json
import datetime
import random
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.supabase import SupabaseVectorStore
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from supabase.client import Client, create_client

# .env 파일에서 환경 변수 로드
load_dotenv()

# 환경 변수 확인
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
TARGET_GITBOOK_NAME = os.getenv("TARGET_GITBOOK_NAME", "해당 Gitbook")
CHAT_HISTORY_FILE = os.getenv("CHAT_HISTORY_FILE", "chat_history.json")

# 추천 질문 목록 - 실제 문서 내용에 맞게 커스터마이징 필요
DEFAULT_SUGGESTED_QUESTIONS = [
    "Gitbook의 주요 기능은 무엇인가요?",
    "문서를 검색하는 방법은 어떻게 되나요?",
    "FETA 프로젝트의 목적은 무엇인가요?",
    "API 문서는 어디서 확인할 수 있나요?",
    "개발자 가이드는 어디에 있나요?",
    "설치 방법을 알려주세요",
    "자주 묻는 질문과 답변"
]

# 문맥별 추천 질문 생성 함수
def generate_context_questions(last_answer, llm):
    try:
        if not last_answer or len(last_answer) < 20:
            return []
        
        prompt = f"""
        다음은 사용자 질문에 대한 답변입니다:
        {last_answer[:500]}...

        위 답변을 기반으로, 사용자가 다음으로 물어볼 만한 관련 질문 3가지를 생성해주세요.
        짧고 명확한 질문으로 작성하세요. 
        JSON 형식 없이 질문만 줄바꿈으로 구분하여 반환하세요.
        """
        
        # LLM으로 질문 생성
        response = llm.invoke(prompt)
        questions = response.content.strip().split('\n')
        
        # 빈 줄 제거하고 앞뒤 공백 제거
        questions = [q.strip() for q in questions if q.strip()]
        
        # 최대 3개 반환
        return questions[:3]
    except Exception as e:
        print(f"추천 질문 생성 오류: {e}")
        return []

# 채팅 내역 저장 함수
def save_chat_history():
    chat_data = {
        "chat_history": st.session_state.chat_history,
    }
    
    # 각 대화 내역도 저장
    for chat_name, chat_id in st.session_state.chat_history:
        chat_key = f"chat_{chat_id}"
        if chat_key in st.session_state:
            chat_data[chat_key] = st.session_state[chat_key]
    
    try:
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"채팅 내역 저장 중 오류 발생: {e}")

# 채팅 내역 불러오기 함수
def load_chat_history():
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                chat_data = json.load(f)
                
            # 채팅 히스토리 목록 불러오기
            if "chat_history" in chat_data:
                st.session_state.chat_history = chat_data["chat_history"]
            
            # 각 대화 내역 불러오기
            for key, value in chat_data.items():
                if key.startswith("chat_") and key != "chat_history":
                    st.session_state[key] = value
    except Exception as e:
        st.warning(f"채팅 내역 불러오기 중 오류 발생: {e}")
        # 오류 발생 시 초기화
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

# 환경 변수 유효성 검사
if not OPENAI_API_KEY or not SUPABASE_URL or not SUPABASE_ANON_KEY:
    missing_vars = []
    if not OPENAI_API_KEY:
        missing_vars.append("OPENAI_API_KEY")
    if not SUPABASE_URL:
        missing_vars.append("SUPABASE_URL")
    if not SUPABASE_ANON_KEY:
        missing_vars.append("SUPABASE_ANON_KEY")
    
    st.error(f"환경 변수가 설정되지 않았습니다: {', '.join(missing_vars)}")
    st.info("create_env.py 스크립트로 생성된 .env 파일을 편집하여 필요한 값을 채워주세요.")
    st.stop()

# Supabase 클라이언트 초기화 (한 번만 실행되도록 캐싱)
@st.cache_resource
def init_supabase_client():
    try:
        return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    except Exception as e:
        st.error(f"Supabase 클라이언트 초기화 실패: {e}")
        return None

# ChatOpenAI 모델 초기화 (한 번만 실행되도록 캐싱)
@st.cache_resource
def init_chat_model():
    return ChatOpenAI(
        temperature=0.1, 
        model_name='gpt-3.5-turbo', 
        openai_api_key=OPENAI_API_KEY
    )

supabase_client = init_supabase_client()
if not supabase_client:
    st.stop()

# LLM 모델 초기화
llm = init_chat_model()

# 채팅 히스토리 로드 (앱 시작 시)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    load_chat_history()

# 추천 질문 초기화
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = random.sample(DEFAULT_SUGGESTED_QUESTIONS, min(4, len(DEFAULT_SUGGESTED_QUESTIONS)))

# 대화 메모리 초기화 (세션 상태 사용)
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

# Langchain 구성 요소 초기화 (한 번만 실행되도록 캐싱)
@st.cache_resource
def init_langchain_components(_supabase_client, _memory): 
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        
        # Supabase Vector Store 초기화
        vector_store = SupabaseVectorStore(
            client=_supabase_client,
            embedding=embeddings,
            table_name="documents",
            query_name="match_documents"
        )
        
        llm = ChatOpenAI(
            temperature=0.1, 
            model_name='gpt-3.5-turbo', 
            openai_api_key=OPENAI_API_KEY
        )
        
        # 검색기 설정
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                'k': 5, 
                'score_threshold': 0.5, 
                'filter': {} 
            } 
        )
        
        # ConversationalRetrievalChain 사용 (대화 기억 기능 포함)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=_memory,
            return_source_documents=True,
            return_generated_question=True,
        )
            
        return qa_chain
    except Exception as e:
        st.error(f"Langchain 구성 요소 초기화 실패: {e}")
        return None

qa_chain = init_langchain_components(supabase_client, st.session_state.memory)
if not qa_chain:
    st.stop()

# 추천 질문 처리 함수
def handle_suggested_question(question):
    # 사용자 질문을 채팅창에 추가
    st.session_state.messages.append({"role": "user", "content": question})
    
    # 답변 생성
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response_content = ""
        
        with st.spinner("답변을 생성 중입니다... 🤔"):
            try:
                # Langchain QA 실행
                response = qa_chain.invoke({"question": question})
                
                # 응답 추출
                answer = response.get("answer", "")
                source_documents = response.get("source_documents", [])
                    
                if not answer:
                    answer = "죄송합니다, 답변을 찾을 수 없습니다. 컨텍스트가 부족하거나 질문이 명확하지 않을 수 있습니다."

                full_response_content = answer

                if source_documents:
                    full_response_content += "\n\n---\n**참고 문서:**\n"
                    # 중복된 source URL을 제거하기 위한 set
                    unique_sources = set()
                    for doc in source_documents:
                        source_url = doc.metadata.get('source', '출처 정보 없음')
                        if source_url not in unique_sources and source_url != '출처 정보 없음':
                            # URL의 마지막 부분을 제목처럼 사용
                            link_title = source_url.split('/')[-1] or source_url.split('/')[-2] or "문서"
                            link_title = link_title.replace('-', ' ').title() # 가독성 향상
                            full_response_content += f"- [{link_title}]({source_url})\n"
                            unique_sources.add(source_url)
                
                message_placeholder.markdown(full_response_content)
                
                # 맥락에 맞는 새로운 추천 질문 생성
                context_questions = generate_context_questions(answer, llm)
                if context_questions:
                    st.session_state.suggested_questions = context_questions
                else:
                    # 새로운 기본 질문 표시
                    st.session_state.suggested_questions = random.sample(
                        DEFAULT_SUGGESTED_QUESTIONS, 
                        min(3, len(DEFAULT_SUGGESTED_QUESTIONS))
                    )

            except Exception as e:
                st.error(f"답변 생성 중 오류가 발생했습니다: {e}")
                full_response_content = "죄송합니다, 현재 답변을 드릴 수 없습니다. 관리자에게 문의해주세요."
                message_placeholder.markdown(full_response_content)
                
                # 오류 발생 시 기본 추천 질문 표시
                st.session_state.suggested_questions = random.sample(
                    DEFAULT_SUGGESTED_QUESTIONS, 
                    min(3, len(DEFAULT_SUGGESTED_QUESTIONS))
                )

        # 메시지 히스토리에 추가
        st.session_state.messages.append({"role": "assistant", "content": full_response_content})
        
        # 대화 자동 저장
        # 현재 대화 이름 저장
        if len(st.session_state.messages) == 3:  # 첫 번째 질문 후 제목 생성
            first_user_msg = question
            chat_title = first_user_msg[:15] + ("..." if len(first_user_msg) > 15 else "")
            st.session_state["current_time_str"] = chat_title
    
    # 페이지 새로고침
    st.rerun()

# --- Streamlit UI ---
st.title("📚 Gitbook Q&A Chatbot")
st.caption(f"'{TARGET_GITBOOK_NAME}' 문서 기반 질문 답변 서비스") 

# 사이드바 설정
st.sidebar.header("챗봇 설정")

# 대화 히스토리 섹션 추가
st.sidebar.markdown("---")
st.sidebar.subheader("💬 대화 히스토리")

# 저장된 대화 히스토리 표시
history_cols = st.sidebar.columns([4, 1])
for i, (chat_name, chat_id) in enumerate(st.session_state.chat_history):
    # 대화 선택
    if history_cols[0].button(f"{chat_name}", key=f"history_{i}", use_container_width=True):
        # 선택한 대화 내용 불러오기
        st.session_state.messages = st.session_state[f"chat_{chat_id}"].copy()
        # 메모리 초기화 (해당 대화에 맞게)
        st.session_state.memory.clear()
        # 메모리 재구성 (대화 내용 기반)
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                user_msg = msg["content"]
            elif msg["role"] == "assistant" and user_msg:
                st.session_state.memory.chat_memory.add_user_message(user_msg)
                st.session_state.memory.chat_memory.add_ai_message(msg["content"])
                user_msg = None
        st.rerun()
    
    # 대화 삭제 버튼
    if history_cols[1].button("🗑️", key=f"delete_{i}", help="이 대화 삭제하기"):
        # 대화 삭제 확인
        chat_key = f"chat_{chat_id}"
        if chat_key in st.session_state:
            del st.session_state[chat_key]
        
        # 히스토리에서 제거
        st.session_state.chat_history.pop(i)
        save_chat_history()
        st.rerun()

# 새 대화 시작 버튼
if st.sidebar.button("➕ 새 대화 시작", use_container_width=True):
    # 현재 대화가 있으면 저장
    if "messages" in st.session_state and len(st.session_state.messages) > 1:
        # 새 대화 ID 생성 (타임스탬프 기반)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        chat_id = f"chat_{timestamp}"
        
        # 대화 제목 생성
        current_title = st.session_state.get("current_time_str", "새 대화")
        
        # 현재 대화 내용 저장
        st.session_state[f"chat_{chat_id}"] = st.session_state.messages.copy()
        st.session_state.chat_history.append((current_title, chat_id))
        save_chat_history()
    
    # 새 대화 시작 - 메모리 초기화
    st.session_state.memory.clear()
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! Gitbook 문서에 대해 무엇이든 물어보세요."}]
    
    # 추천 질문 초기화
    st.session_state.suggested_questions = random.sample(
        DEFAULT_SUGGESTED_QUESTIONS, 
        min(4, len(DEFAULT_SUGGESTED_QUESTIONS))
    )
    
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info(
    "이 챗봇은 FETA Gitbook 문서 내용을 기반으로 답변합니다."
)

# 채팅 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! Gitbook 문서에 대해 무엇이든 물어보세요."}]

# 이전 채팅 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 추천 질문 표시 (첫 메시지 또는 마지막 메시지가 assistant인 경우)
if len(st.session_state.messages) == 1 or st.session_state.messages[-1]["role"] == "assistant":
    # 추천 질문 컨테이너
    with st.container():
        st.write("#### 추천 질문:")
        cols = st.columns(len(st.session_state.suggested_questions))
        
        for i, question in enumerate(st.session_state.suggested_questions):
            if cols[i].button(question, key=f"suggested_{i}", use_container_width=True):
                handle_suggested_question(question)

# 사용자 입력
if prompt := st.chat_input("질문을 입력해주세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response_content = ""
        
        with st.spinner("답변을 생성 중입니다... 🤔"):
            try:
                # Langchain QA 실행 (ConversationalRetrievalChain)
                response = qa_chain.invoke({"question": prompt})
                
                # 응답 추출
                answer = response.get("answer", "")
                source_documents = response.get("source_documents", [])
                    
                if not answer:
                    answer = "죄송합니다, 답변을 찾을 수 없습니다. 컨텍스트가 부족하거나 질문이 명확하지 않을 수 있습니다."

                full_response_content = answer

                if source_documents:
                    full_response_content += "\n\n---\n**참고 문서:**\n"
                    # 중복된 source URL을 제거하기 위한 set
                    unique_sources = set()
                    for doc in source_documents:
                        source_url = doc.metadata.get('source', '출처 정보 없음')
                        if source_url not in unique_sources and source_url != '출처 정보 없음':
                            # URL의 마지막 부분을 제목처럼 사용
                            link_title = source_url.split('/')[-1] or source_url.split('/')[-2] or "문서"
                            link_title = link_title.replace('-', ' ').title() # 가독성 향상
                            full_response_content += f"- [{link_title}]({source_url})\n"
                            unique_sources.add(source_url)
                
                message_placeholder.markdown(full_response_content)
                
                # 맥락에 맞는 새로운 추천 질문 생성
                context_questions = generate_context_questions(answer, llm)
                if context_questions:
                    st.session_state.suggested_questions = context_questions
                else:
                    # 새로운 기본 질문 표시
                    st.session_state.suggested_questions = random.sample(
                        DEFAULT_SUGGESTED_QUESTIONS, 
                        min(3, len(DEFAULT_SUGGESTED_QUESTIONS))
                    )

            except Exception as e:
                st.error(f"답변 생성 중 오류가 발생했습니다: {e}")
                full_response_content = "죄송합니다, 현재 답변을 드릴 수 없습니다. 관리자에게 문의해주세요."
                message_placeholder.markdown(full_response_content)
                
                # 오류 발생 시 기본 추천 질문 표시
                st.session_state.suggested_questions = random.sample(
                    DEFAULT_SUGGESTED_QUESTIONS, 
                    min(3, len(DEFAULT_SUGGESTED_QUESTIONS))
                )

        st.session_state.messages.append({"role": "assistant", "content": full_response_content})
        
        # 대화 자동 저장
        # 현재 대화 이름 저장
        if len(st.session_state.messages) == 3:  # 첫 번째 질문 후 제목 생성
            first_user_msg = prompt
            chat_title = first_user_msg[:15] + ("..." if len(first_user_msg) > 15 else "")
            st.session_state["current_time_str"] = chat_title

# 채팅 기록 지우기 버튼
st.sidebar.markdown("---")
all_cols = st.sidebar.columns([1, 1])

if all_cols[0].button("모든 대화 지우기", use_container_width=True):
    # 대화 메모리 초기화
    st.session_state.memory.clear()
    # 대화 히스토리 초기화
    st.session_state.chat_history = []
    # 현재 대화 초기화
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! Gitbook 문서에 대해 무엇이든 물어보세요."}]
    # 저장된 모든 대화 삭제
    for key in list(st.session_state.keys()):
        if key.startswith("chat_"):
            del st.session_state[key]
    # 파일 삭제 (선택 사항)
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            os.remove(CHAT_HISTORY_FILE)
        except:
            pass
    # 추천 질문 초기화
    st.session_state.suggested_questions = random.sample(
        DEFAULT_SUGGESTED_QUESTIONS, 
        min(4, len(DEFAULT_SUGGESTED_QUESTIONS))
    )
    st.rerun()

# 현재 대화 저장 버튼
if all_cols[1].button("대화 저장하기", use_container_width=True):
    # 직접 현재 대화 저장
    if "messages" in st.session_state and len(st.session_state.messages) > 1:
        # 타임스탬프 기반 ID 생성
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        chat_id = f"chat_{timestamp}"
        
        # 대화 제목 생성
        current_title = st.session_state.get("current_time_str", "새 대화")
        if not current_title or current_title == "새 대화":
            current_title = f"대화 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # 현재 대화 내용 저장
        st.session_state[f"chat_{chat_id}"] = st.session_state.messages.copy()
        st.session_state.chat_history.append((current_title, chat_id))
        save_chat_history()
        st.success("대화가 저장되었습니다!")
    else:
        st.warning("저장할 대화가 없습니다.")