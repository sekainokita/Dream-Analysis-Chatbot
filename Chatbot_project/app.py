# 환경 구성
import os
import sys
import streamlit as st
import openai
from dotenv import load_dotenv

# LangChain 주요 구성
import langchain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS as CommunityFAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate


# --------------------------------------
# (1) .env 파일에서 OPENAI_API_KEY 로드
# --------------------------------------
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    st.error("OpenAI API 키가 설정되지 않았습니다.")
    st.stop()

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = API_KEY


# --------------------------------------
# (2) FAISS 벡터스토어 로드 함수
# --------------------------------------
def load_vectorstore():
    """저장된 FAISS 벡터스토어를 로드"""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        r"../Chatbot_project/db/faiss",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


# --------------------------------------
# (3) RAG 체인 설정 함수
# --------------------------------------
def setup_rag_chain(system_prompt: str):
    """시스템 프롬프트 + 로드된 FAISS 벡터스토어로 ConversationalRetrievalChain 설정"""
    vectorstore = load_vectorstore()
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    #model="gpt-4o-mini",
    #model="gpt-3.5-turbo",
    #model="gpt-4o",
    #model="gpto1",

    # 검색된 문서와 질문을 기반으로 최종 답변을 생성하는 프롬프트 정의
    qa_prompt = PromptTemplate.from_template("""
    아래는 시스템 프롬프트입니다:
    {system_prompt}
    당신은 [지그문트 프로이트]의 [정신분석학]과 [카를 융]의 [심리역동학]에 큰 영향을 받은 [신경정신과 의사] 입니다. 
    제가 꿈을 말하면 [프로이트의 꿈 분석학]과 [카를 융의 심리역동학]을 바탕으로 분석해주세요. 
    반드시 다음과 같은 규칙을 준수하세요.
    [규칙] :
    0. 당신의 역할을 들키면 안됩니다.
    0-1. 저는 당신의 병원에 방문한 환자라고 생각해주세요.
    1. 사용자의 질의에 철저히 심리 분석 지식을 활용해 대답합니다. 
    2. 입력받은 꿈 프롬프트에 대해 키워드별로 분석해야합니다.
    3. 심리역동적 해석을 포함해서 치료적인 제언을 추가해야 합니다.
    4. 꿈 프롬프트와 관련하여 다시 되묻는 과정을 추가해야 합니다.
    5. 결과 근거가 부족한 부분은 원문을 다시 확인해야 한다는 가이드라인과 함께 최대한 신중하게 대답합니다.
    
    다음은 검색을 통해 얻은 문서 조각들입니다:
    {context}

    사용자의 질문:
    {question}

    위 정보를 바탕으로, 반드시 한국어 만을 사용해서, 사용자에게 심리학적 분석과 치료적 제언을 포함한 답변을 작성하세요.
    당신은 [지그문트 프로이트]의 [정신분석학]과 [카를 융]의 [심리역동학]에 큰 영향을 받은 [신경정신과 의사] 입니다. 
    제가 꿈을 말하면 [프로이트의 꿈 분석학]과 [카를 융의 심리역동학]을 바탕으로 분석해주세요. 
    반드시 다음과 같은 규칙을 준수하세요.
    [규칙] :
    0. 당신의 역할을 들키면 안됩니다.
    0-1. 저는 당신의 병원에 방문한 환자라고 생각해주세요.
    1. 사용자의 질의에 철저히 심리 분석 지식을 활용해 대답합니다. 
    2. 입력받은 꿈 프롬프트에 대해 키워드별로 분석해야합니다.
    3. 심리역동적 해석을 포함해서 치료적인 제언을 추가해야 합니다.
    4. 꿈 프롬프트와 관련하여 다시 되묻는 과정을 추가해야 합니다.
    5. 결과 근거가 부족한 부분은 원문을 다시 확인해야 한다는 가이드라인과 함께 최대한 신중하게 대답합니다.
    """)

    # ConversationalRetrievalChain 생성
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=False,  # 참조 문서를 반환하지 않음
        verbose=True
    )

    return qa_chain


# --------------------------------------
# (4) 시스템 프롬프트 정의
# --------------------------------------
SYSTEM_PROMPT = """당신은 [지그문트 프로이트]의 [정신분석학]과 [카를 융]의 [심리역동학]에 큰 영향을 받은 [신경정신과 심리상담가] 입니다. 
제가 꿈을 말하면 [프로이트의 꿈 분석학]과 [카를 융의 심리역동학]을 바탕으로 분석해주세요. 
반드시 다음과 같은 규칙을 준수하세요.
[규칙] :
0. 당신의 역할을 들키면 안됩니다.
0-1. 저는 당신의 병원에 방문한 환자라고 생각해주세요.
1. 사용자의 질의에 철저히 심리 분석 지식을 활용해 대답합니다. 
2. 입력받은 꿈 프롬프트에 대해 [반드시] [키워드]별로 분석해야합니다.
3. 심리역동적 해석을 포함해서 치료적인 제언을 추가해야 합니다.
4. 꿈 프롬프트와 관련하여 다시 되묻는 과정을 추가해야 합니다.
5. 결과 근거가 부족한 부분은 원문을 다시 확인해야 한다는 가이드라인과 함께 최대한 신중하게 대답합니다.
"""

if "qa_chain" not in st.session_state:
    # 초기화 시점에 RAG 체인 세팅
    st.session_state["qa_chain"] = setup_rag_chain(SYSTEM_PROMPT)

if "chat_history" not in st.session_state:
    # ConversationalRetrievalChain에서 사용하는 (질문, 답변) 튜플 기록
    st.session_state["chat_history"] = []


# --------------------------------------
# (5) Streamlit 앱 본문
# --------------------------------------
st.title("😴 무의식의 나")
st.write("프로이트의 정신분석학과 카를 융의 심리역동학을 학습한 AI가 꿈을 해석해드립니다.")

# ------------------------------
# 5-1) 사용자와 어시스턴트 메시지만 출력
# ------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "안녕하세요. 먼저 편안히 앉으시죠. 당신의 꿈에 대해 자세히 들어보고 싶습니다."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ------------------------------
# 5-2) 사용자 입력 처리 및 RAG 실행
# ------------------------------
if prompt := st.chat_input("🔎당신의 꿈을 이야기해주세요..."):
    # (1) 입력 메시지 저장 및 출력
    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)
    st.chat_message("user").write(prompt)

    # (2) RAG 체인을 통해 답변 생성
    rag_chain = st.session_state["qa_chain"]
    result = rag_chain({
        "question": prompt,
        "chat_history": st.session_state["chat_history"],
        "system_prompt": SYSTEM_PROMPT  # system_prompt 전달
    })
     
    # (3) 어시스턴트 메시지 저장 및 출력
    assistant_msg = result["answer"]
    st.session_state.messages.append({"role": "assistant", "content": assistant_msg})
    st.chat_message("assistant").write(assistant_msg)

    # (4) 대화 히스토리 갱신 (질문, 답변 형태로 저장)
    st.session_state["chat_history"].append((prompt, assistant_msg))


# ------------------------------
# 5-3) UI 구성 요소 추가 (사이드바)
# ------------------------------
if st.sidebar.button("대화 초기화"):
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.rerun()

if st.sidebar.checkbox("시스템 프롬프트 보기"):
    st.sidebar.text_area("현재 시스템 프롬프트", SYSTEM_PROMPT, height=300)
