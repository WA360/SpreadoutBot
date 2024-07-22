# flask
from flask import Flask, jsonify, request, Response, stream_with_context

from flask_cors import CORS

from dotenv import load_dotenv
import os
import json
import re

# bedrock chatbot 필요 라이브러리
import boto3

# langchain buffer memory , langchain history 라이브러리
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from operator import itemgetter
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import (
    RunnableLambda,
    ConfigurableFieldSpec,
    RunnablePassthrough,
)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.exceptions import OutputParserException

# langchain pdf reader , pdf splitter
from langchain_community.document_loaders import PyPDFLoader
import pymupdf as fitz
from langchain_text_splitters import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# langchain embedding 필요 라이브러리
from langchain_community.embeddings import BedrockEmbeddings
from chromadb.utils import embedding_functions
from langchain_huggingface import HuggingFaceEmbeddings

# lanchain bedrodk 라이브러리
from langchain_aws import ChatBedrock
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.output_parsers import (
    StructuredOutputParser,
    ResponseSchema,
    PydanticOutputParser
)

# langchain 리트리버 라이브러리
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# chromadb
from langchain_chroma import Chroma
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings

# mysql
from flask_mysqldb import MySQL


app = Flask(__name__)
CORS(app)

s3_bucket_name = os.getenv("AWS_BUCKET")
s3_region = os.getenv("AWS_REGION")
s3_access_key_id = os.getenv("S3_ACCESS_KEY")
s3_secret_access_key = os.getenv("S3_SECRET_ACCESS_KEY")
bedrock_region = os.getenv("BEDROCK_AWS_REGION")
bedrock_access_key_id = os.getenv("BEDROCK_ACCESS_KEY")
bedrock_secret_access_key = os.getenv("BEDROCK_SECRET_ACCESS_KEY")
mysql_host = os.getenv("MYSQL_HOST")
mysql_user = os.getenv("MYSQL_USER")
mysql_password = os.getenv("MYSQL_PASSWORD")
mysql_db = os.getenv("MYSQL_DB")
host = os.getenv("HOST")

database_client = None  # db 위치
embedding = None  # 임베딩 방법
llm = None  # llm
retriever = None  # 검색기
conversation = None  # 채팅 버퍼
store = {}  # 채팅 기록?
# mysql = None  # mysql 디비
app.config["MYSQL_HOST"] = mysql_host
app.config["MYSQL_USER"] = mysql_user
app.config["MYSQL_PASSWORD"] = mysql_password
app.config["MYSQL_DB"] = mysql_db
mysql = MySQL(app)

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=bedrock_region,
    aws_access_key_id=bedrock_access_key_id,
    aws_secret_access_key=bedrock_secret_access_key,
)


class Document:
    def __init__(self, metadata, page_content):
        self.metadata = metadata
        self.page_content = page_content


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []


class Summary(BaseModel):
    summary: str = Field(description="summarize what is given")
    keywords: List[str] = Field(description="List of 5 important keywords for given content")


# 기본 세팅
def setCahtBot():
    setDB(host)
    setEmbedding("bedrock")
    setLLM()


def setDB(loc):
    global database_client
    if loc == "local":
        database_client = chromadb.PersistentClient(
            path="./chroma_data",
            settings=Settings(),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE,
        )
    elif loc == "server":
        database_client = chromadb.HttpClient(
            host="localhost",
            # host="host.docker.internal",
            port=7000,
            ssl=False,
            headers=None,
            settings=Settings(allow_reset=True),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE,
        )
    elif loc == "ec2":
        print(f"loc :{loc}")
        database_client = chromadb.HttpClient(
            host="chroma",
            # host="host.docker.internal",
            # host="localhost",
            port=8000,
            ssl=False,
            headers=None,
            settings=Settings(allow_reset=True),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE,
        )


def setEmbedding(loc):
    global embedding
    if loc == "chroma":
        embedding = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    elif loc == "langchain":
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    elif loc == "bedrock":
        embedding = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v1", client=bedrock
        )


def setLLM():
    global llm
    llm = ChatBedrock(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        # model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        client=bedrock,
        streaming=True,
    )


@app.route("/save/pdf", methods=["POST"])
def setPdf():
    data = request.get_json()
    fileName = data["fileName"]
    fileNum = data["fileNum"]
    chapterId = data["chapterId"]
    bucket_name = s3_bucket_name
    download_path = f"./pdfs/{fileName}"

    # file download from S3
    s3 = boto3.client(
        "s3",
        aws_access_key_id=s3_access_key_id,
        aws_secret_access_key=s3_secret_access_key,
        region_name=s3_region,
    )

    try:
        # PDF 파일 다운로드
        s3.download_file(bucket_name, fileName, download_path)
        print(f"Downloaded {fileName} from bucket {bucket_name} to {download_path}")
    except Exception as e:
        print(f"Error downloading file: {e}")
    print("파일 다운로드함")
    # pdf load
    def check_file_exists_in_pdfs(filename):
        return os.path.isfile(f"./pdfs/{filename}")
    is_file = check_file_exists_in_pdfs(fileName)
    if is_file:
        print("파일 다운 성공")
        # fileName = "jotcoding-1-25.pdf"
        # fileNum = 18
        # fileName = "jotcoding.pdf"
        download_path = f"./pdfs/{fileName}"
        pdf_document = fitz.open(download_path)
        # db 연결
        chroma_db = Chroma(
            client=database_client,
            collection_name=f"{fileNum}.pdf",
            embedding_function=embedding,
        )
        print(f"{chroma_db._collection.count()}개 있음")
        if chroma_db._collection.count() == 0:
            # 목차 읽기
            toc = pdf_document.get_toc()

            # 각 단원의 시작 및 끝 페이지를 저장할 리스트
            chapters = []

            print(f"목차 길이: {len(toc)}")
            for i in range(len(toc) - 1):
                current_chapter = toc[i]
                next_chapter = toc[i + 1]

                chapter_level = current_chapter[0]
                chapter_title = current_chapter[1]
                start_page = current_chapter[2] - 1  # 페이지 번호는 0부터 시작
                end_page = next_chapter[2] - 2  # 다음 챕터 시작 전까지 포함
                chapters.append((chapter_title, start_page, end_page))
                print(chapter_title, start_page, end_page)
            # 마지막 챕터 추가
            last_chapter = toc[-1]
            chapter_title = last_chapter[1]
            start_page = last_chapter[2] - 1
            end_page = pdf_document.page_count - 1
            chapters.append((chapter_title, start_page, end_page))
            print(f"마지막 챕터 {chapter_title}, {start_page}, {end_page}")

            # # 각 단원의 내용을 배열에 저장
            chapter_contents = []
            ori_chapter_contents = []
            print("----------------------내용 추출 시작-----------------------------")
            print(f"챕터 길이: {len(chapters)}")
            for chapter in chapters:
                title, start_page, end_page = chapter
                content = ""
                print(f"<<{title}>> 내용 추출 진행중")
                if start_page > end_page:
                    end_page = start_page
                for page_num in range(start_page, end_page + 1):
                    page = pdf_document.load_page(page_num)
                    content += page.get_text()
                ori_chapter_contents.append(
                    Document(
                        metadata={"title": title},
                        page_content=str(content),
                    )
                )    
                new_content = process_text(content)
                chapter_contents.append(
                    Document(
                        metadata={"title": title},
                        page_content=str(new_content),
                    )
                )
                # print(
                #     "시작------------------------------------------------------------------------------------------"
                # )
                # print(f"title: {title}")
                # print(f"content: {content}")
                # print(
                #     "끝------------------------------------------------------------------------------------------"
                # )

            # save to db
            Chroma.from_documents(
                # documents=docs,
                documents=chapter_contents,
                embedding=embedding,
                collection_name=f"{fileNum}.pdf",
                client=database_client,
            )

            
            # --------------------------------------------------------------------
            # 파서 설정
            # # pydantic_parser
            output_parser = PydanticOutputParser(pydantic_object=Summary)
            # StructuredOutputParser
            # response_schemas = [
            #     ResponseSchema(name="summary", description="Summarize the given content in Markdown format in 300 characters or less."),
            #     ResponseSchema(
            #         name="keywords",
            #         description="List of 5 important keywords for given content",
            #     ),
            # ]
            # output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
            format_instructions =output_parser.get_format_instructions()
            # 프롬프트 설정
            system_prompt = (
                "당신은 인문학적 영역에 전문가인 도우미 입니다."
                "주어진 내용을 사용하여 질문에 답하세요. summary,keywords를 제외하고는 한글로 답하세요"
                "주어진 정보에 대한 답변이 없을 경우, 알고 있는 대로 답변해 주십시오."
                "반드시 json 포맷으로 응답하세요."
                "\n\n"
                "{format_instructions}"
            )
            final_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    (
                        "human",
                        " \n\n {content}\n \n위의 내용 중에서 {title}이 가리키는 부분을 찾아 내용을 요약하고 중요 키워드를 5개 뽑아주세요.반드시 json 포맷으로 응답하세요, key는 summary와 keywords 를 사용하세요.",
                    ),
                ]
            )
            # llm 및 체인 설정
            llm = ChatBedrock(
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                client=bedrock,
                streaming=True,
            )
            chain = final_prompt | llm | output_parser
            # --------------------------------------------------------------------------
        
            cur = mysql.connection.cursor()
            query = (
                "update api_chapter ac set ac.summary =%s ,ac.keywords=%s where ac.id=%s"
            )
            print("----------------------요약 및 키워드 추출 시작---------------------------")
            cnt=1
            i = 0
            while i<  len(ori_chapter_contents):
                chapter = ori_chapter_contents[i]
                print(f"{round((cnt/len(ori_chapter_contents))*100,2)}% 진행중 <<{chapter.metadata["title"]}>>")

                try:
                    response = chain.invoke(
                        {
                            "content": chapter.page_content, 
                            "title": chapter.metadata["title"],
                            "format_instructions":format_instructions
                        }
                    )
                    # print(response)
                    summary_match = re.search(r"summary='(.*?)' keywords=", str(response), re.DOTALL)
                    keywords_match = re.search(r"keywords=\[(.*?)\]", str(response), re.DOTALL)
                    if summary_match and keywords_match:
                        summary = summary_match.group(1)
                        keywords = keywords_match.group(1).replace("'", "").split(", ")

                        # JSON 객체 생성
                        data = {
                            "summary": summary,
                            "keywords": keywords
                        }

                        # JSON 객체를 문자열로 변환
                        response = json.dumps(data, ensure_ascii=False, indent=4)
                    # print(response)
                    if is_json(response)==False:
                        print("json 아니라서 다시함")
                        continue
                    response = json.loads(response)
                    summary = response["summary"]
                    keywords = response["keywords"]
                    list_as_string = json.dumps(keywords, ensure_ascii=False)
                    # print(f"title: {chapter.metadata["title"]}")
                    # print(f"Summary: {summary}")
                    # print(f"Keywords: {keywords}")
                    # print(f"chapterId: {chapterId}")
                    rrr = cur.execute(query, (summary, list_as_string, chapterId))
                    # print(rrr)
                except OutputParserException as e:
                    print(f"아웃풋 파서 에러")
                    print(f"OutputParserException: {e}")
                    continue
                except json.JSONDecodeError as e:
                    print(f"JSON Decode Error: {e}")
                except KeyError as e:
                    print(f"Key Error: Missing key {e}")
                chapterId += 1
                cnt+=1
                i+=1
                # print("끝----------------------------------------------------------------")
            mysql.connection.commit()
            # db 연결 종료
            cur.close()
            # pdf 종료
            pdf_document.close()
            
            os.remove(download_path)
            return jsonify({"result": "upload success"})
        else:
            pdf_document.close()
            os.remove(download_path)
            return jsonify({"result": "file already exists"})
    else:
        return jsonify({"result": "not found file"})


@app.route("/test", methods=["GET"])
def testtest():
    print("테스트 데스와~")
    return jsonify({"result": "테스트 데스와~"})


@app.route("/question/langchain", methods=["POST"])
def mtest3():
    global store
    data = request.get_json()
    if "fileNum" in data:
        fileNum = data["fileNum"]
        chatNum = data["chatNum"]
        chat_name = f"{fileNum}_{chatNum}"
        userQuestion = data["question"]
        print("userQuestion: ", userQuestion)
        # 리트리버 세팅
        chroma_db = Chroma(
            client=database_client,
            collection_name=f"{fileNum}.pdf",
            embedding_function=embedding,
        )
        retriever = chroma_db.as_retriever(search_kwargs={"k": 30})

        # 히스토리 프롬프트
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        # 히스토피 프롬프트 합체
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # 히스토리 리트리버 합체
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        system_prompt = (
            # "You are an assistant for question-answering tasks. "
            "당신은 인문학적 영역에 전문가인 도우미 입니다."
            "주어진 내용을 사용하여 질문에 답하세요. 반드시 한글로 답하세요"
            "주어진 정보에 대한 답변이 없을 경우, 알고 있는 대로 답변해 주십시오."
            "answer in detail and use markdown"
            # "'책' 라는 단어가 있으면 주어진 내용에서만 답을 하세요."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # 채팅 기록?

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in store:
                # history = getHistory(session_id)
                # store[session_id] = history
                store[session_id] = ChatMessageHistory()
            return store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        # # # 그냥 답변
        # res = conversational_rag_chain.invoke(
        #     {"input": userQuestion},
        #     config={
        #         "configurable": {"session_id": chat_name}
        #     },  # constructs a key "abc123" in `store`.
        # )["answer"]

        # result = []
        # for message in store[chat_name].messages:
        #     if isinstance(message, AIMessage):
        #         prefix = "AI"
        #     else:
        #         prefix = "User"
        #     result.append({prefix: f"{message.content}\n"})

        # # 저장소 출력
        # # updateresult = updateHistory(store[chat_name], chatNum)
        # # print(updateresult)
        # print(store[chat_name])
        # return jsonify({"result": res})
        # return jsonify({"result": result})

        # 스트림 답변
        def generate():
            # messages = [HumanMessage(content=userQuestion)]
            for chunk in conversational_rag_chain.stream(
                {"input": userQuestion},
                config={
                    "configurable": {"session_id": chat_name}
                },  # constructs a key "abc123" in `store`.
            ):
                # yield f"{chunk.content}\n"
                if isinstance(chunk, dict) and "answer" in chunk:
                    # print(chunk)
                    yield chunk["answer"]
                # print(chunk.content, end="|", flush=True)

        # # 저장소 출력
        # print(store)

        return Response(stream_with_context(generate()), content_type="text/event-stream")
    else:
        userQuestion = data["question"]
        if userQuestion:
            system_prompt = (
                "당신은 인류학적 영역을 잘 알고 있는 도우미 입니다. 반드시 한글로 답하세요."
                "\n\n"
                # "{context}"
            )
            final_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    (
                        "human",
                        "{question}",
                    ),
                ]
            )

            chain = final_prompt | llm
            # response = chain.invoke({"question": userQuestion})
            # return response.content
            def generate():
                # messages = [HumanMessage(content=userQuestion)]
                for chunk in chain.stream({"question": userQuestion}):
                    # yield f"{chunk.content}\n"
                    yield chunk.content
            return Response(
                stream_with_context(generate()), content_type="text/event-stream"
            )
        else:
            return jsonify({"result": "question 없음"})



@app.route("/question/langchain/test", methods=["POST"])
def sendQuestionBylangchain():
    data = request.get_json()
    fileName = data["fileName"]
    fileNum = data["fileNum"]
    userQuestion = data["question"]
    print(userQuestion)
    # load from disk
    chroma_db = Chroma(
        client=database_client,
        collection_name=f"{fileNum}.pdf",
        embedding_function=embedding,
    )
    # docs = chroma_db.similarity_search(question, k=2)
    # print(docs)
    retriever = chroma_db.as_retriever(search_kwargs={"k": 10})

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a helpful assistant.
                Answer questions using only the following context.
                If you don't know the answer just say you don't know,
                don't makel it up:
                \n\n
                {context}
                """,
            ),
            ("human", "{question}"),
        ]
    )

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    # result = chain.invoke(userQuestion)
    result = chain.invoke([HumanMessage(content=userQuestion)])

    return jsonify({"result": f"{result.content}"})

    # def generate():
    #     # messages = [HumanMessage(content=userQuestion)]
    #     for chunk in chain.stream(userQuestion):
    #         # yield f"{chunk.content}\n"
    #         yield chunk.content
    #         # print(chunk.content, end="|", flush=True)
    # return Response(stream_with_context(generate()), content_type="text/event-stream")




# 텍스트 임베딩 함수
def get_embeddings(text, bedrock_client):
    response = bedrock_client.invoke_model(
        modelId="bedrock-embedding", body=text, contentType="application/json"
    )
    embeddings = response["body"]["embedding"]
    return embeddings


# 텍스트 요약 함수
def summarize_text(text):
    llm2 = ChatBedrock(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        client=bedrock,
        streaming=True,
    )
    summary = llm2.invoke(
        text + "\n\n 위의 내용을 원본에 가깝게 요약해줘"
    )
    # print(summary.content)
    return summary.content


# 주어진 텍스트를 8000 토큰 이하로 요약하는 함수
def process_text(text):
    current_text = text
    # token_count = count_tokens(current_text)
    token_count = llm.get_num_tokens(current_text)
    while token_count > 7000:
        print(f"token_count: {token_count}")
        current_text = summarize_text(current_text)
        token_count = llm.get_num_tokens(current_text)
        print(f"줄인 token_count: {token_count}")

    return current_text, token_count

# 파일 있는지 체크
def check_file_exists_in_pdfs(filename):
    return os.path.isfile(f"./pdfs/{filename}")

def is_json(obj):
    try:
        json_object = json.loads(obj)
        # { } 가 포함된 string이 invalid json 인 경우 Exception
        iterator = iter(json_object)
        # { } 가 없는 경우는 string의 경우 Exception
    except Exception as e:
        return False
    return True

# def getHistory(sessionId):
#     url = f"http://localhost:3000/bot/session/detail?chapterId={sessionId}"
#     response = requests.get(url).json()
#     # print(response[0]["content"])
#     return response[0]["content"]
#     return jsonify({"result": response[0]["content"]})


# # @app.route("/test/h", methods=["GET"])
# def updateHistory(content, chapterId):
#     data = {"content": content, "chapterId": chapterId}
#     url = f"http://localhost:3000/bot/session/detail"
#     response = requests.put(url, data=data)
#     return response


#     # def generate():
#     #     # messages = [HumanMessage(content=userQuestion)]
#     #     for chunk in chain.stream(userQuestion):
#     #         # yield f"{chunk.content}\n"
#     #         yield chunk.content
#     #         # print(chunk.content, end="|", flush=True)
#     # return Response(stream_with_context(generate()), content_type="text/event-stream")

# def checkbcrypt():
#     msg = "총, 균, 쇠 (재레드 다이아몬드) (Z-Library).pdf"
#     msg = msg.encode('utf-8')
#     print('저장된 해시값:', msg)
#     # bytes_password = b"총, 균, 쇠 (재레드 다이아몬드) (Z-Library).pdf" #// 비밀번호
#     bytes_hashed_password = bcrypt.hashpw(password=msg, salt=bcrypt.gensalt())
#     print('저장된 해시값:', bytes_hashed_password)

# checkbcrypt()

# 챗봇 기본 세팅
setCahtBot()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3100)
