# flask
from flask import Flask, jsonify, request, Response, stream_with_context

from flask_cors import CORS

from dotenv import load_dotenv
import os

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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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

# langchain embedding 필요 라이브러리
from langchain_community.embeddings import BedrockEmbeddings
from chromadb.utils import embedding_functions
from langchain_huggingface import HuggingFaceEmbeddings

# lanchain bedrodk 라이브러리
from langchain_aws import ChatBedrock
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# langchain 리트리버 라이브러리
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.messages import AIMessage

# chromadb
from langchain_chroma import Chroma
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings


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
host = os.getenv("HOST")

database_client = None  # db 위치
embedding = None  # 임베딩 방법
llm = None  # llm
retriever = None  # 검색기
conversation = None  # 채팅 버퍼
store = {}  # 채팅 기록?
chat_memory = ConversationBufferMemory(human_prefix="Human", ai_prefix="Assistant")

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
    setEmbedding("langchain")
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
            # host="localhost",
            host="host.docker.internal",
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
            host="localhost",
            port=7000,
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
            model_id="amazon.titan-embed-text-v2:0", client=bedrock
        )


def setLLM():
    global llm
    llm = ChatBedrock(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
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


# def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
#     if (user_id, conversation_id) not in store:
#         store[(user_id, conversation_id)] = InMemoryHistory()
#     return store[(user_id, conversation_id)]

# def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = InMemoryHistory()
#     return store[session_id]


# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = InMemoryChatMessageHistory()
#     return store[session_id]


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

            for page_num in range(start_page, end_page + 1):
                page = pdf_document.load_page(page_num)
                content += page.get_text()

            chapter_contents.append(
                Document(
                    # metadata={title: title, page: start_page}, page_content=content
                    metadata={title: title},
                    page_content=content,
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


# @app.route("/question/langchain", methods=["POST"])
# def sendQuestionBylangchain():
#     data = request.get_json()
#     fileName = data["fileName"]
#     fileNum = data["fileNum"]
#     collection_name = f"{fileNum}_{fileName}"
#     userQuestion = data["question"]
#     print(userQuestion)
#     # load from disk
#     chroma_db = Chroma(
#         client=database_client,
#         collection_name=collection_name,
#         embedding_function=embedding,
#     )
#     # docs = chroma_db.similarity_search(question, k=2)
#     # print(docs)
#     retriever = chroma_db.as_retriever(search_kwargs={"k": 30})

#     prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 """
#                 You are a helpful assistant.
#                 Answer questions using only the following context.
#                 If you don't know the answer just say you don't know,
#                 don't makel it up:
#                 \n\n
#                 {context}
#                 """,
#             ),
#             ("human", "{question}"),
#         ]
#     )

#     chain = (
#         {
#             "context": retriever,
#             "question": RunnablePassthrough(),
#         }
#         | prompt
#         | llm
#     )

#     result = chain.invoke(userQuestion)

#     return jsonify({"result": f"{result.content}"})

#     # def generate():
#     #     # messages = [HumanMessage(content=userQuestion)]
#     #     for chunk in chain.stream(userQuestion):
#     #         # yield f"{chunk.content}\n"
#     #         yield chunk.content
#     #         # print(chunk.content, end="|", flush=True)
#     # return Response(stream_with_context(generate()), content_type="text/event-stream")


@app.route("/question/bedrock", methods=["POST"])
def sendQuestionByBedrock():
    # userQuestion = request.args.get("question")
    data = request.get_json()
    userQuestion = data["question"]
    # print("userQuestion: ", userQuestion)

    if userQuestion:
        # body <- Inference configuration
        # body = {
        #     "anthropic_version": "bedrock-2023-05-31",
        #     "max_tokens": 1000,
        #     "messages": [
        #         {
        #             "role": "user",
        #             "content": [
        #                 {
        #                     "type": "text",
        #                     "text": userQuestion,
        #                 },
        #             ],
        #         }
        #     ],
        # }
        # # invoke_model <- API Request
        # response = bedrock.invoke_model(
        #     modelId="anthropic.claude-3-haiku-20240307-v1:0",
        #     contentType="application/json",
        #     accept="application/json",
        #     body=json.dumps(body),
        # )
        # status_code = response["ResponseMetadata"]["HTTPStatusCode"]
        # if status_code == 200:
        #     response_body = json.loads(response["body"].read())
        #     return jsonify({"result": response_body["content"][0]["text"]})
        # else:
        #     return jsonify({"result": "뭔가 에러남"})

        # response = bedrock.invoke_model_with_response_stream(
        #     body=json.dumps(body),
        #     modelId="anthropic.claude-3-haiku-20240307-v1:0",
        #     accept="application/json",
        #     contentType="application/json",
        # )
        # status_code = response["ResponseMetadata"]["HTTPStatusCode"]
        # if status_code == 200:
        #     return Response(stream_llm_response(response), content_type="text/plain")
        # else:
        #     return jsonify({"result": "뭔가 에러남"})

        def generate():
            messages = [HumanMessage(content=userQuestion)]
            for chunk in llm.stream(messages):
                # yield f"{chunk.content}\n"
                yield chunk.content

        return Response(
            stream_with_context(generate()), content_type="text/event-stream"
        )
    else:
        return jsonify({"result": userQuestion + " 없음"})


def check_file_exists_in_pdfs(filename):
    return os.path.isfile(f"./pdfs/{filename}")


@app.route("/test", methods=["GET"])
def testtest():
    print("테스트 데스와~")
    return jsonify({"result": "테스트 데스와~"})


# 챗봇 기본 세팅
setCahtBot()


@app.route("/test/m", methods=["POST"])
def mtest():
    data = request.get_json()
    userQuestion = data["question"]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You're an assistant who's good at {ability}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    chain = prompt | llm

    chain_with_history = RunnableWithMessageHistory(
        chain,
        # Uses the get_by_session_id function defined in the example
        # above.
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
        history_factory_config=[
            ConfigurableFieldSpec(
                id="session_id",
                annotation=str,
                name="Session ID",
                description="Unique identifier for the session.",
                default="",
                is_shared=True,
            ),
        ],
    )

    # result = []

    # result.append(
    #     chain_with_history.invoke(  # noqa: T201
    #         {"ability": "math", "question": "What does cosine mean?"},
    #         config={"configurable": {"session_id": "foo"}},
    #     )
    # )
    # result = chain_with_history.invoke(  # noqa: T201
    #     {"ability": "math", "question": "What does cosine mean?"},
    #     config={"configurable": {"session_id": "foo"}},
    # )
    result = chain_with_history.invoke(  # noqa: T201
        {"ability": "llm", "question": userQuestion},
        config={"configurable": {"session_id": "foo"}},
    )

    print(result.content)

    # Uses the store defined in the example above.
    # result.append(store)  # noqa: T201

    # result.append(
    #     chain_with_history.invoke(  # noqa: T201
    #         {"ability": "math", "question": "What's its inverse"},
    #         config={"configurable": {"session_id": "foo"}},
    #     )
    # )

    # result.append(store)  # noqa: T201

    return jsonify({"result": f"{result.content}"})


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


# @app.route("/test/m3", methods=["POST"])
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
    retriever = chroma_db.as_retriever(search_kwargs={"k": 3})

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
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
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
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    ## 그냥 답변
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

    return jsonify({"result": result})

    ## 스트림 답변
    # def generate():
    #     # messages = [HumanMessage(content=userQuestion)]
    #     for chunk in conversational_rag_chain.stream(
    #         {"input": userQuestion},
    #         config={
    #             "configurable": {"session_id": chat_name}
    #         },  # constructs a key "abc123" in `store`.
    #     ):
    #         # yield f"{chunk.content}\n"
    #         if isinstance(chunk, dict) and "answer" in chunk:
    #             # print(chunk)
    #             yield chunk["answer"]
    #         # print(chunk.content, end="|", flush=True)

    # return Response(stream_with_context(generate()), content_type="text/event-stream")


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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
