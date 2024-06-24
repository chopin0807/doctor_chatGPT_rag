import pandas as pd
import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_qa_with_sources_chain
from langchain.chains import ConversationalRetrievalChain

total = pd.read_csv("그룹핑 결과.csv", encoding = "utf-8")
# 존재 교수/의사명에 대한 리스트 정리 및 출력
exist = []
for i in total["언급 의사/교수"]:
    if i not in exist:
        exist.append(i)
print("존재 교수/의사명 리스트: ", exist)
# 교수/의사명 질의 => 예) ㄱㄱㄱ교수, ㄱㅈㅁ원장
query = input("질문대상 교수/의사 이름(예시:ㄱㄱㄱ교수, ㄱㄷㄱ원장):")

# chatGPT rag 방식을 위한 text파일 작성
f = open("temp.txt", "w", encoding = "utf-8")
for a, b, c, d in zip(total["언급 의사/교수"], total["본문제목"], total["본내용"], total["댓글"]):
    target = ""
    if a == query:
        if b != "X":
            target += b
        if c != "X":
            target += c
        if d != "X":
            target += d
        target += "\n"
    f.write(target)
f.close()

# rag방식 진행
try: # 해당 검색내용이 있는지 여부 확인
    # open ai에서 발급받은 api key를 등록
    api_key = "<발급받은 api key 입력>"

    # 키워드 문서 로딩
    loader = TextLoader('temp.txt', encoding = 'utf-8')
    data = loader.load()

    # 키워드 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300, chunk_overlap = 50, length_function = len)
    document = text_splitter.split_documents(data)
    # 저장
    embedding_model = OpenAIEmbeddings(api_key=api_key)
    db = Chroma.from_documents(document, embedding_model)

    # 지식 검색
    retriever = db.as_retriever()
    # 응답 생성
    llm_src = ChatOpenAI(temperature = 0, model = "gpt-3.5-turbo", api_key = api_key)
    qa_chain = create_qa_with_sources_chain(llm_src)
    retrieval_qa = ConversationalRetrievalChain.from_llm(llm_src, retriever)

    # 질의응답 출력
    output = retrieval_qa({
        "question": "'{}'에 대해 언급된 내용을 종합하여 1)'{}'의 친절도에 대한 점수를 100점만점 기준으로 제시하고, 2)'{}'의 치료 성향 및 어떤 수술을 주로 진행하는에 대한 정보가 있다면 요약해주고, 3)'{}'의 수술 예후에 대한 정보가 있다면 요약해줘. 이에 대해서 1) 2) 3) 번호를 매개면서 정리해줘.".format(query, query, query, query),
        "chat_history": []
    })

    print(f"답변: {output['answer']}")
except:
    print("답변: chromadb까지 모든 라이브러리(import 라이브러리)를 pip install로 설치해야 하거나, openai api 서비스를 이용할 수 없거나(온라인 연결 및 api_key 사용가능한지 확인), 문서에서 {}에 관한 언급 없음".format(query))
os.remove("temp.txt")