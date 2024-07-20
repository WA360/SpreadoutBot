# flask
from flask import Flask, jsonify, request, Response, stream_with_context

from flask_cors import CORS

from dotenv import load_dotenv
import os
import json

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
    ResponseSchema,
    StructuredOutputParser,
    PydanticOutputParser,
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


class Document:
    def __init__(self, metadata, page_content):
        self.metadata = metadata
        self.page_content = page_content


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


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []


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
        # model_id="anthropic.claude-3-haiku-20240307-v1:0",
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        client=bedrock,
        streaming=True,
    )


def extract_chapters(file_path):
    # PDF 파일 열기
    pdf_document = fitz.open(file_path)

    # 목차 읽기
    toc = pdf_document.get_toc()

    # 각 단원의 시작 및 끝 페이지를 저장할 리스트
    chapters = []

    for i in range(len(toc) - 1):
        current_chapter = toc[i]
        next_chapter = toc[i + 1]

        chapter_title = current_chapter[1]
        start_page = current_chapter[2] - 1  # 페이지 번호는 0부터 시작
        end_page = next_chapter[2] - 2  # 다음 챕터 시작 전까지 포함

        chapters.append((chapter_title, start_page, end_page))

    # 마지막 챕터 추가
    last_chapter = toc[-1]
    chapter_title = last_chapter[1]
    start_page = last_chapter[2] - 1
    end_page = pdf_document.page_count - 1

    chapters.append((chapter_title, start_page, end_page))

    # 각 단원의 내용을 배열에 저장
    chapter_contents = []

    for chapter in chapters:
        title, start_page, end_page = chapter
        content = ""

        for page_num in range(start_page, end_page + 1):
            page = pdf_document.load_page(page_num)
            content += page.get_text()

        chapter_contents.append((title, content))

    return chapter_contents


@app.route("/save/pdf", methods=["POST"])
def setPdf():
    data = request.get_json()
    fileName = data["fileName"]
    fileNum = data["fileNum"]
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

    # pdf load
    is_file = check_file_exists_in_pdfs(fileName)
    if is_file:
        print("파일 다운로드 완료!")
        collection_name = f"{fileNum}_{fileName}"

        # # 기본 나누기(청크 단위로 나누기)
        # documents = PyPDFLoader(download_path).load_and_split()
        # documents = PyPDFLoader("./pdfs/Chapters.pdf").load_and_split()
        # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # docs = text_splitter.split_documents(documents)
        # print(docs)
        # 새로운 나누기(챕터별로 나누기)--------------------------------------------------
        # PDF 파일 열기
        pdf_document = fitz.open(download_path)

        # 목차 읽기
        toc = pdf_document.get_toc()

        # 각 단원의 시작 및 끝 페이지를 저장할 리스트
        chapters = []

        for i in range(len(toc) - 1):
            current_chapter = toc[i]
            next_chapter = toc[i + 1]

            chapter_title = current_chapter[1]
            start_page = current_chapter[2] - 1  # 페이지 번호는 0부터 시작
            end_page = next_chapter[2] - 2  # 다음 챕터 시작 전까지 포함

            chapters.append((chapter_title, start_page, end_page))

        # 마지막 챕터 추가
        last_chapter = toc[-1]
        chapter_title = last_chapter[1]
        start_page = last_chapter[2] - 1
        end_page = pdf_document.page_count - 1

        chapters.append((chapter_title, start_page, end_page))

        # 각 단원의 내용을 배열에 저장
        chapter_contents = []

        for chapter in chapters:
            title, start_page, end_page = chapter
            content = ""
            print(f"title: {title} 진행중")
            for page_num in range(start_page, end_page + 1):
                page = pdf_document.load_page(page_num)
                content += page.get_text()
            new_content = process_text(content)
            chapter_contents.append(
                Document(
                    # metadata={title: title, page: start_page}, page_content=content
                    metadata={"title": title},
                    page_content=str(new_content),
                )
            )
        # --------------------------------------------------------------------
        chroma_db = Chroma(
            client=database_client,
            collection_name=collection_name,
            embedding_function=embedding,
        )
        print(f"{chroma_db._collection.count()}개 있음")
        pdf_document.close()
        if chroma_db._collection.count() == 0:
            # save to disk
            Chroma.from_documents(
                # documents=docs,
                documents=chapter_contents,
                embedding=embedding,
                collection_name=collection_name,
                client=database_client,
            )
            os.remove(download_path)
            return jsonify({"result": "upload success"})
        else:
            os.remove(download_path)
            return jsonify({"result": "file already exists"})
    else:
        return jsonify({"result": "파일 없음, 다운로드실패?"})


@app.route("/save/pdf/test", methods=["POST"])
def setPdftest():
    # fileName = "jotcoding-1-25.pdf"
    fileName = "csapp13.pdf"
    download_path = f"./pdfs/{fileName}"

    pdf_document = fitz.open(download_path)

    # 목차 읽기
    toc = pdf_document.get_toc()

    # 각 단원의 시작 및 끝 페이지를 저장할 리스트
    chapters = []

    for i in range(len(toc) - 1):
        current_chapter = toc[i]
        next_chapter = toc[i + 1]

        chapter_level = current_chapter[0]
        chapter_title = current_chapter[1]
        start_page = current_chapter[2] - 1  # 페이지 번호는 0부터 시작
        end_page = next_chapter[2] - 2  # 다음 챕터 시작 전까지 포함
        # print(f"{chapter_level} ,{chapter_title} , {start_page} , {end_page}")
        chapters.append((chapter_title, start_page, end_page))

    # 마지막 챕터 추가
    last_chapter = toc[-1]
    chapter_title = last_chapter[1]
    start_page = last_chapter[2] - 1
    end_page = pdf_document.page_count - 1

    chapters.append((chapter_title, start_page, end_page))

    # # 각 단원의 내용을 배열에 저장
    chapter_contents = []

    for chapter in chapters:
        title, start_page, end_page = chapter
        content = ""
        if start_page > end_page:
            end_page = start_page
        for page_num in range(start_page, end_page + 1):
            page = pdf_document.load_page(page_num)
            content += page.get_text()

        # print(
        #     "시작------------------------------------------------------------------------------------------"
        # )
        # print(f"title: {title}")
        # print(f"content: {content}")
        # print(
        #     "끝------------------------------------------------------------------------------------------"
        # )

        chapter_contents.append(
            Document(
                # metadata={title: title, page: start_page}, page_content=content
                metadata={"title": title, "page": start_page},
                page_content=content,
            )
        )
    # --------------------------------------------------------------------
    # 프롬프트 설정
    system_prompt = (
        "당신은 컴퓨터 사이언스를 잘 알고 있는 도우미 입니다."
        "주어진 내용을 사용하여 질문에 답하세요. 반드시 한글로 답하세요"
        "주어진 정보에 대한 답변이 없을 경우, 알고 있는 대로 답변해 주십시오."
        "반드시 json 포맷으로 응답하세요. key 는summary 와 keywords 를 사용하세요"
        "\n\n"
        "{context}"
    )
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "{title}이 가리키는 부분을 찾아 내용을 요약하고 중요 키워드를 5개 뽑아주세요.",
            ),
        ]
    )

    # llm 및 체인 설정
    llm = ChatBedrock(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        client=bedrock,
        streaming=True,
    )
    # chain = final_prompt | llm | output_parser
    chain = final_prompt | llm
    # --------------------------------------------------------------------------
    # print(format_instructions)
    chapterId = 2625
    cur = mysql.connection.cursor()
    query = "update api_chapter ac set ac.summary =%s ,ac.keywords=%s where ac.id=%s"
    for chapter in chapter_contents:
        print(f"chapter:{chapterId}-----------------------------------------------")
        response = chain.invoke(
            {"context": chapter.page_content, "title": chapter.metadata["title"]}
        )
        try:
            data = json.loads(response.content)
            # print(data)
            summary = data["summary"]
            keywords = data["keywords"]
            # list_as_string = json.dumps(keywords, ensure_ascii=False)
            print(f"Summary: {summary}")
            print(f"Keywords: {keywords}")
            # print(f"Chapter ID: {chapterId}")
            # cur.execute(query, (summary, list_as_string, chapterId))
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
        except KeyError as e:
            print(f"Key Error: Missing key {e}")
        chapterId += 1
        # print("끝----------------------------------------------------------------")
    mysql.connection.commit()
    cur.close()

    # return jsonify({"result": chapters})
    # return jsonify({"result": "upload success"})
    # chroma_db = Chroma(
    #     client=database_client,
    #     collection_name=collection_name,
    #     embedding_function=embedding,
    # )
    # print(f"{chroma_db._collection.count()}개 있음")
    # pdf_document.close()
    # if chroma_db._collection.count() == 0:
    #     # save to disk
    #     Chroma.from_documents(
    #         # documents=docs,
    #         documents=chapter_contents,
    #         embedding=embedding,
    #         collection_name=collection_name,
    #         client=database_client,
    #     )
    #     os.remove(download_path)
    #     return jsonify({"result": "upload success"})
    # else:
    #     os.remove(download_path)
    #     return jsonify({"result": "file already exists"})
    return jsonify({"result": "file already exists"})


@app.route("/save/pdf/test2", methods=["POST"])
def setPdftest2():
    print("요청은 왔음")
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
    is_file = check_file_exists_in_pdfs(fileName)
    if is_file:
        print("파일 다운 성공")
        # fileName = "jotcoding-1-25.pdf"
        # fileNum = 18
        # fileName = "jotcoding.pdf"
        download_path = f"./pdfs/{fileName}"
        collection_name = f"{fileNum}_{fileName}"
        pdf_document = fitz.open(download_path)

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
            new_content = process_text(content)
            chapter_contents.append(
                Document(
                    # metadata={title: title, page: start_page}, page_content=content
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

        # --------------------------------------------------------------------
        # 프롬프트 설정
        system_prompt = (
            "당신은 인문학적 영역에 전문가인 도우미 입니다."
            "주어진 내용을 사용하여 질문에 답하세요. 반드시 한글로 답하세요"
            "주어진 정보에 대한 답변이 없을 경우, 알고 있는 대로 답변해 주십시오."
            "반드시 json 포맷으로 응답하세요. key 는summary 와 keywords 를 사용하세요"
            "\n\n"
            "{context}"
        )
        final_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "human",
                    "{title}이 가리키는 부분을 찾아 내용을 요약하고 중요 키워드를 5개 뽑아주세요.",
                ),
            ]
        )

        # llm 및 체인 설정
        llm = ChatBedrock(
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            client=bedrock,
            streaming=True,
        )
        chain = final_prompt | llm
        # --------------------------------------------------------------------------
    
        cur = mysql.connection.cursor()
        query = (
            "update api_chapter ac set ac.summary =%s ,ac.keywords=%s where ac.id=%s"
        )
        print("----------------------요약 및 키워드 추출 시작---------------------------")
        for chapter in chapter_contents:
            print(f"<<{chapter.metadata["title"]}>> 요약 및 키워드 추출 진행중----------")
            response = chain.invoke(
                {"context": chapter.page_content, "title": chapter.metadata["title"]}
            )
            try:
                data = json.loads(response.content)
                # print(data)
                summary = data["summary"]
                keywords = data["keywords"]
                list_as_string = json.dumps(keywords, ensure_ascii=False)
                print(f"title: {chapter.metadata["title"]}")
                print(f"Summary: {summary}")
                print(f"Keywords: {keywords}")
                cur.execute(query, (summary, list_as_string, chapterId))
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error: {e}")
            except KeyError as e:
                print(f"Key Error: Missing key {e}")
            chapterId += 1
            # print("끝----------------------------------------------------------------")
        mysql.connection.commit()
        cur.close()

        chroma_db = Chroma(
            client=database_client,
            collection_name=collection_name,
            embedding_function=embedding,
        )
        print(f"{chroma_db._collection.count()}개 있음")
        pdf_document.close()
        if chroma_db._collection.count() == 0:
            # save to disk
            Chroma.from_documents(
                # documents=docs,
                documents=chapter_contents,
                embedding=embedding,
                collection_name=collection_name,
                client=database_client,
            )
            os.remove(download_path)
            return jsonify({"result": "upload success"})
        else:
            os.remove(download_path)
            return jsonify({"result": "file already exists"})
    else:
        return jsonify({"result": "not found file"})


@app.route("/pdf/checktoc", methods=["GET"])
def pdfToc():
    # fileName = "jotcoding-1-25.pdf"
    fileNum = 7
    fileName = "UTILITARIANISM.pdf"
    data = request.args["book"]
    if data:
        fileName = data
    print(fileName)
    download_path = f"./pdfs/{fileName}"
    pdf_document = fitz.open(download_path)

    # 목차 읽기
    toc = pdf_document.get_toc()
    print(f"목차 개수: {len(toc)}")
    # 각 단원의 시작 및 끝 페이지를 저장할 리스트
    if len(toc) > 0:
        chapters = []
        for i in range(len(toc) - 1):
            current_chapter = toc[i]
            next_chapter = toc[i + 1]

            chapter_level = current_chapter[0]
            chapter_title = current_chapter[1]
            start_page = current_chapter[2] - 1  # 페이지 번호는 0부터 시작
            end_page = next_chapter[2] - 2  # 다음 챕터 시작 전까지 포함
            # print(f"{chapter_level} ,{chapter_title} , {start_page} , {end_page}")
            chapters.append((chapter_title, start_page, end_page))
            print(chapter_title, start_page, end_page)
        # # 마지막 챕터 추가
        # last_chapter = toc[-1]
        # chapter_title = last_chapter[1]
        # start_page = last_chapter[2] - 1
        # end_page = pdf_document.page_count - 1

        # chapters.append((chapter_title, start_page, end_page))

        # # 각 단원의 내용을 배열에 저장
        chapter_contents = []
        print(f"챕터 길이: {len(chapters)}")
        for chapter in chapters:
            title, start_page, end_page = chapter
            content = ""
            if start_page > end_page:
                end_page = start_page
            for page_num in range(start_page, end_page + 1):
                page = pdf_document.load_page(page_num)
                content += page.get_text()
            new_content = process_text(content)
            chapter_contents.append(
                Document(
                    # metadata={title: title, page: start_page}, page_content=content
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

        # 요약작업 시작
        # # --------------------------------------------------------------------
        # # 프롬프트 설정
        # system_prompt = (
        #     "당신은 컴퓨터 사이언스를 잘 알고 있는 도우미 입니다."
        #     "주어진 내용을 사용하여 질문에 답하세요. 반드시 한글로 답하세요"
        #     "주어진 정보에 대한 답변이 없을 경우, 알고 있는 대로 답변해 주십시오."
        #     "반드시 json 포맷으로 응답하세요. key 는summary 와 keywords 를 사용하세요"
        #     "\n\n"
        #     "{context}"
        # )
        # final_prompt = ChatPromptTemplate.from_messages(
        #     [
        #         ("system", system_prompt),
        #         (
        #             "human",
        #             "{title}이 가리키는 부분을 찾아 내용을 요약하고 중요 키워드를 5개 뽑아주세요.",
        #         ),
        #     ]
        # )

        # # llm 및 체인 설정
        # llm = ChatBedrock(
        #     model_id="anthropic.claude-3-haiku-20240307-v1:0",
        #     client=bedrock,
        #     streaming=True,
        # )
        # # chain = final_prompt | llm | output_parser
        # chain = final_prompt | llm
        # # --------------------------------------------------------------------------
        # # print(format_instructions)
        # chapterId = 1
        # for chapter in chapter_contents:
        #     print(f"chapter:{chapterId}-----------------------------------------------")
        #     response = chain.invoke(
        #         {"context": chapter.page_content, "title": chapter.metadata["title"]}
        #     )
        #     try:
        #         data = json.loads(response.content)
        #         # print(data)
        #         summary = data["summary"]
        #         keywords = data["keywords"]
        #         # list_as_string = json.dumps(keywords, ensure_ascii=False)
        #         print(f"title: {chapter.metadata["title"]}")
        #         print(f"Summary: {summary}")
        #         print(f"Keywords: {keywords}")
        #         # print(f"Chapter ID: {chapterId}")
        #         # cur.execute(query, (summary, list_as_string, chapterId))
        #     except json.JSONDecodeError as e:
        #         print(f"JSON Decode Error: {e}")
        #     except KeyError as e:
        #         print(f"Key Error: Missing key {e}")
        #     chapterId += 1
        #     # print("끝----------------------------------------------------------------")
        return jsonify({"result": "데스와~"})
    else:
        return jsonify({"result": "toc 없음"})


@app.route("/question/langchain/test", methods=["POST"])
def sendQuestionBylangchain():
    data = request.get_json()
    fileName = data["fileName"]
    fileNum = data["fileNum"]
    collection_name = f"{fileNum}_{fileName}"
    userQuestion = data["question"]
    print(userQuestion)
    # load from disk
    chroma_db = Chroma(
        client=database_client,
        collection_name=collection_name,
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


def check_file_exists_in_pdfs(filename):
    return os.path.isfile(f"./pdfs/{filename}")


@app.route("/test", methods=["GET"])
def testtest():
    print("테스트 데스와~")
    return jsonify({"result": "테스트 데스와~"})


@app.route("/test/m2", methods=["POST"])
def mtest2():
    with_message_history = RunnableWithMessageHistory(llm, get_session_history)

    config = {"configurable": {"session_id": "abc2"}}
    response = with_message_history.invoke(
        [HumanMessage(content="Hi! I'm Bob")],
        config=config,
    )
    print(response.content)
    response = with_message_history.invoke(
        [HumanMessage(content="What's my name?")],
        config=config,
    )
    print(response.content)
    config = {"configurable": {"session_id": "abc3"}}

    response = with_message_history.invoke(
        [HumanMessage(content="What's my name?")],
        config=config,
    )
    print(response.content)
    config = {"configurable": {"session_id": "abc2"}}

    response = with_message_history.invoke(
        [HumanMessage(content="What's my name?")],
        config=config,
    )

    print(response.content)
    print("__________________________________________________________________________")
    print(get_session_history("abc2"))
    print("__________________________________________________________________________")

    return jsonify({"result": "test"})


@app.route("/question/langchain/prompt", methods=["POST"])
def mtest4():
    global store
    data = request.get_json()
    fileName = data["fileName"]
    fileNum = data["fileNum"]
    chatNum = data["chatNum"]
    history_prompt = None
    retriever_prompt = None
    if "history_prompt" in data:
        history_prompt = data["history_prompt"]
    if "retriever_prompt" in data:
        retriever_prompt = data["retriever_prompt"]
    collection_name = f"{fileNum}_{fileName}"
    chat_name = f"{fileNum}_{fileName}_{chatNum}"
    userQuestion = data["question"]
    print("collection_name: ", collection_name)
    print("userQuestion: ", userQuestion)
    # 리트리버 세팅
    chroma_db = Chroma(
        client=database_client,
        collection_name=collection_name,
        embedding_function=embedding,
    )
    retriever = chroma_db.as_retriever(search_kwargs={"k": 3})
    # 히스토리 프롬프트
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    if history_prompt:
        contextualize_q_system_prompt = history_prompt
    # print(f"contextualize_q_system_prompt: {contextualize_q_system_prompt}")
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
        "You are a helpful assistant"
        "Use the following pieces of retrieved context to answer the question."
        "If there is no answer to the given information, answer what you know."
        "answer in detail and use markdown"
        "'책' 라는 단어가 있으면 주어진 내용에서만 답을 하세요."
        "\n\n"
        "{context}"
    )
    if retriever_prompt:
        system_prompt = retriever_prompt + "\n\n{context}"
    # print(f"system_prompt: {system_prompt}")
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

    # 그냥 답변
    conversational_rag_chain.invoke(
        {"input": userQuestion},
        config={
            "configurable": {"session_id": chat_name}
        },  # constructs a key "abc123" in `store`.
    )["answer"]

    result = []
    for message in store[chat_name].messages:
        if isinstance(message, AIMessage):
            prefix = "AI"
        else:
            prefix = "User"
        result.append({prefix: f"{message.content}\n"})

    # 저장소 출력
    # updateresult = updateHistory(store[chat_name], chatNum)
    # print(updateresult)
    print(store[chat_name])
    return jsonify({"result": result})

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


@app.route("/question/langchain", methods=["POST"])
def mtest3():
    global store
    data = request.get_json()
    fileName = data["fileName"]
    fileNum = data["fileNum"]
    chatNum = data["chatNum"]
    collection_name = f"{fileNum}_{fileName}"
    chat_name = f"{fileNum}_{fileName}_{chatNum}"
    userQuestion = data["question"]
    print("collection_name: ", collection_name)
    print("userQuestion: ", userQuestion)
    # 리트리버 세팅
    chroma_db = Chroma(
        client=database_client,
        collection_name=collection_name,
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
        "You are a helpful assistant"
        "Use the following pieces of retrieved context to answer the question."
        "If there is no answer to the given information, answer what you know."
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

    # # 그냥 답변
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
        text + "\n\n 위의 내용 8000토큰 보다 작지만 원본에 가깝게 요약해줘"
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

# file_path = "./pdfs/csapp13.pdf"
# chapter_contents = extract_chapters(file_path)

# for title, content in chapter_contents:
#     print(f"Chapter: {title}")
#     print(content)
#     print(
#         "--------------------------------------------------------------------------------------"
#     )


# 챗봇 기본 세팅
setCahtBot()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3100)
