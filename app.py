import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import hashlib

# (deprecatd) from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings

# (deprecatd) from langchain.vectorstores import FAISS # 문서검색을 담당하는 페이스북이 만든 고속 Vector DB. 로컬에 설치하며, 프로그램 종료시 DB는 삭제됨
from langchain_community.vectorstores import FAISS

from htmlTemplates import css, bot_template, user_template   # htmlTemplates.py 파일안에 있는 모듈들을 가져옴

def get_pdf_text(pdf_docs):
    # 빈 text 배열 생성
    text = ""
    for pdf in pdf_docs:
        # 페이지별로 배열을 리턴해주는 PdfReader클래스의 객체 생성
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            # 각 페이지의 텍스트를 추출하여 모든 text를 배열로 저장 --> extract_text 매소드를 이용
            text += page.extract_text()
    return text

def get_text_chunk(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    ##############################
    #     embedding API 선택      #
    ##############################

    # 선택 1: OpenAI embedding API 사용시 (유료)
    embeddings = OpenAIEmbeddings()

    # 선택 2: 허깅페이스에서 제공하는 Instructor embedding API 사용시 (무료)
    # 성능은 OpenAIEmbeddings보다 우수하지만 느리다.
    # https://huggingface.co/hkunlp/instructor-xl
    # model_name 인자값으로 hkunlp/instructor-xl을 입력
    # 단, 2개의 dependency를 설치해야 함 pip install instructorembedding sentence_transformers
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# get_conversation_chain(vectorscore) 실행에 필요한 모듈
# (deprecated) from langchain.chat_models import ChatOpenAI # ConversationBufferMemory()의 인자로 들어갈 llm으로 ChatOpenAI모델을 사용하기로 함
from langchain_community.chat_models import ChatOpenAI

from langchain.memory import ConversationBufferMemory # 대화내용을 저장하는 memory
from langchain.chains import ConversationalRetrievalChain

######################################################
#  핵심 함수 GCC : get_conversation_chain(vectorscore) #
######################################################
# 입력 : VectorDB
# 출력 : 질문을 입력하면 DB 검색과 결과출력을 담당하는 ConversationalRetrievalChain.from_llm의 객체를 생성하여 반환함
def get_conversation_chain(vectorscore):
   # 메모리에 로드된 env파일에서 "OPENAI_API_KEY"라고 명명된 값을 변수로 저장
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # 선택 1: 대화에 사용될 llm API 객체를 llm 변수에 저장
    llm = ChatOpenAI(
        temperature=0.1,    # 창의성 (0.0 ~ 2.0)
        model_name="gpt-4-turbo-preview", # chatGPT-4 Turbo 사용
        openai_api_key=OPENAI_API_KEY # Automatically inferred from env var OPENAI_API_KEY if not provided.
        )

    # 선택 2: HuggingFaceHub를 llm 모델로 사용시
    # from langchain.llms import HuggingFaceHub
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    # ConverstaionBufferMemory 클래스를 이용하여 대화내용을 chat_history라는 key값으로 저장해주는 memory 객체를 생성
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # llm 객체, memory 객체를 인자로 입력하여 DB 검색결과를 출력하는 ConversationalRetrievalChain.from_llm 객체를 생성
    conversation_chain = ConversationalRetrievalChain.from_llm(
        # 검색을 실행하는 llm 선택
        llm=llm,
        # 검색을 당하는 vector DB를 retriver 포맷으로 저장
        retriever=vectorscore.as_retriever(),
        # 사용자와 대화내용을 메모리에 저장하여 같은 맥락에서 대화를 유지
        memory=memory
    )
    # ConversationalRetrievalChain.from_llm 객체로 생성된 convestaion_chain을 반환
    return conversation_chain

######################################################
#              핵심 함수 handle_userinput              #
######################################################
# user에 받은 질문을 인자로 넣으면 대답을 반환하는 함수
# 핵심함수 GCC에 의해 생성된 st.session_sate.converstaion 객체의 메소드를 활용하여 대답을 생성함
def handle_userinput(user_question) :
    # 질문을 입력하면 DB 검색과 결과출력을 담당하는 ConversationalRetrievalChain.from_llm의 객체로 생성된 것이
    # st.sessioin_state.conversation
    # 따라서 객체의 메소드로서 사용자의 질문은 ({'question': user_question}) 형태로 인자에 넣어주면 결과를 출력하고, 대화내용은 메모리에 저장된다.
    # response['chat_history']에는 사용자의 질문과 대답이 저장되어 있다.
    response = st.session_state.conversation({'question': user_question}) # 객체의 내장메소드에 사용자 질문을 받는 기능이 있을 것임
    # st.write(response) # 딕셔너리로 출력되며 chat_history라는 key값에 질의/응답이 저장되어 있음을 알수있다.

    # 'chat_history'를 key값으로 하여 이번의 질의응답만 저장되어 있는데, 이를 메모리에 누적에서 보관하여 전체 대화를 기록함
    st.session_state.chat_history = response['chat_history']

    # message 객체의 content 속성에 대화가 들어있으므로 이를 추출하여 탬플릿의 {{MSG}} 위치에 넣는 replace 메소드를 사용
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

######################################################
#                    admin key 검증                   #
######################################################

def is_admin(_input_key):
    # 문자열을 byte열로 encoding을 먼저 실시한 후, sha256으로 암호화
    input_key_hash = hashlib.sha256(_input_key.encode()).hexdigest()
    saved_key_hash = hashlib.sha256(os.getenv("OPENAI_API_KEY").encode()).hexdigest()
    if input_key_hash == saved_key_hash :
        return True

###############################
#          (참고)  채팅창        #
###############################
# 사전에 정의한 css, html양식을 st.wirte() 함수의 인자로 넣어주면 웹사이트 형식으로 출력한다.
# st.write(user_template.replace("{{MSG}}", "Hellow Bot"), unsafe_allow_html=True)
# st.write(bot_template.replace("{{MSG}}", "Hellow Human"), unsafe_allow_html=True)

######################################################
#                        Main                        #
######################################################

def main() :
    load_dotenv()
    st.set_page_config(page_title="TONchat", page_icon=":books:", layout="wide")
    # css, html관련 설정은 실제 대화관련 함수보다 앞에서 미리 실행해야 한다.
    st.write(css, unsafe_allow_html=True)


    ###############################
    #             초기화            #
    ###############################

    # st.session_state.conversation = get_conversation_chain(vectorstore)을 통해
    # sesstion_state 객체의 속성으로 conversation이 신설되고, 그 안에 딕셔너리로 질의/응답이 저장된다.
    # { question : ddd, answer : ddd } 이런식이다.
    # 이러한 저장이 이뤄지도록 일단 conversation 속성에 None으로 초기화를 시켜 준비해놓는다.
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None

    ###############################
    #             질문창            #
    ###############################
    # 질문입력창
    st.header("TONchat")
    st.write("Ask a question about Tokamak Network's services")
    st.write("- Titan L2 Network")
    st.write("")
    user_question = st.text_input("Input your question")
    # 질문이 저장되면 if문이 true가 되고, 질문에 대한 답변을 처리한다.
    if user_question:
        handle_userinput(user_question)


    ###############################
    #      sidebar 파일 업로드       #
    ###############################
    with st.sidebar:
        # Admin login
        with st.popover("Admin login"):
            st.markdown("Admin key 🔑")
            admin = is_admin(st.text_input("Input your admin key"))
        if admin :
            st.write("Hi, Admin !")
            logout = st.button("Logout", type="primary")
            st.header("Your documents")
            # upload multiple documents
            pdf_docs = st.file_uploader("Upload your PDFs here and click on 'process'", accept_multiple_files=True)
            if st.button("Process") :
                with st.spinner('Processing') :
                    ########################
                    #      get pdf text    #
                    ########################
                    raw_text = get_pdf_text(pdf_docs)

                    ########################
                    #  get the text chunks #
                    ########################
                    text_chunks = get_text_chunk(raw_text)
                    st.write(text_chunks)

                    ########################
                    #  create vector store #
                    ########################
                    vectorstore = get_vectorstore(text_chunks)

                    ########################################
                    #  핵심함수를 이용한 conversation chain 생성 #
                    ########################################
                    # 핵심함수 get_conversation_chain() 함수를 사용하여, 첫째, 이전 대화내용을 읽어들이고, 둘째, 다음 대화 내용을 반환할 수 있는 객체를 생성
                    # 다만 streamlit 환경에서는 input이 추가되거나, 사용자가 버튼을 누르거나 하는 등 새로운 이벤트가 생기면 코드 전체를 다시 읽어들임
                    # 이 과정에서 변수가 전부 초기화됨.
                    # 따라서 이러한 초기화 및 생성이 반복되면 안되고 하나의 대화 세션으로 고정해주는 st.sessiion_state 객체안에 대화를 저장해야 날아가지 않음
                    # conversation이라는 속성을 신설하고 그 안에 대화내용을 key, value 쌍으로 저장 (딕셔너리 자료형)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
            else:
                st.write("You are not admin")

if __name__ == "__main__":
    main()