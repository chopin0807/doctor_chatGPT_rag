import ollama
import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from multiprocessing import Pool
import pandas as pd

# 한국어 llama3구축 (llama-3-Korean-Bllossom-8B)참고 url => https://bab-dev-study.tistory.com/m/67
total = pd.read_csv("그룹핑 결과.csv", encoding = "utf-8")
# 존재 교수/의사명에 대한 리스트 정리 및 출력
exist = []
for i in total["언급 의사/교수"]:
    if i not in exist:
        exist.append(i)

def llama3_multiprocessing(x):
    # llama3 rag 방식을 위한 text파일 작성
    f = open("temp.txt", "w", encoding = "utf-8")
    for a, b, c, d in zip(total["언급 의사/교수"], total["본문제목"], total["본내용"], total["댓글"]):
        target = ""
        if a == x:
            if b != "X":
                target += b
            if c != "X":
                target += c
            if d != "X":
                target += d
            target += "\n"
        f.write(target)
    f.close()

    # 1. 임시 텍스트 파일(temp.txt)불러오기
    loader = TextLoader("temp.txt", encoding = 'utf-8')
    docs = loader.load()
    os.remove("temp.txt")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20, length_function = len)
    splits = text_splitter.split_documents(docs)
    # 2. Ollama 임베딩 진행
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    # 3. 수동으로 설치한 Ollama모델(llama3-korean)불러오기
    def ollama_llm(question, context):
        formatted_prompt = f"Question: {question}\n\nContext: {context}"
        response = ollama.chat(model='llama3-korean', messages=[{'role': 'user', 'content': formatted_prompt}])
        return response['message']['content']
    # 4. RAG 설정
    retriever = vectorstore.as_retriever()

    def combine_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def rag_chain(question):
        retrieved_docs = retriever.invoke(question)
        formatted_context = combine_docs(retrieved_docs)
        return ollama_llm(question, formatted_context)
        
    # 의사이름 키워드 질의 진행
    print(rag_chain("'{}'에 대해 언급된 내용을 종합하여 1)'{}'의 친절도에 대한 점수를 100점만점 기준으로 제시하고, 2)'{}'의 치료 성향 및 어떤 수술을 주로 진행하는에 대한 정보가 있다면 요약해주고, 3)'{}'의 수술 예후에 대한 정보가 있다면 요약해줘. 이에 대해서 1) 2) 3) 번호를 매개면서 정리해줘.".format(x, x, x, x)))
    print("=" * 100 + x)

    return [x, rag_chain("'{}'에 대해 언급된 내용을 종합하여 1)'{}'의 친절도에 대한 점수를 100점만점 기준으로 제시하고, 2)'{}'의 치료 성향 및 어떤 수술을 주로 진행하는에 대한 정보가 있다면 요약해주고, 3)'{}'의 수술 예후에 대한 정보가 있다면 요약해줘. 이에 대해서 1) 2) 3) 번호를 매개면서 정리해줘.".format(x, x, x, x))]

if __name__ == "__main__":
    with Pool(processes = 10) as pool:
        results = pool.map(llama3_multiprocessing, exist)

    doctor = [i[0] for i in results]
    content = [i[1] for i in results]
    result = pd.DataFrame({"의사명":doctor, "요약내용":content})
    result.to_csv("의사별 요약내용(llama3-kor).csv", encoding="utf-8", index=False)