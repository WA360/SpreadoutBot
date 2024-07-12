# flask
from flask import Flask, jsonify, request, Response, stream_with_context

from flask_cors import CORS

from dotenv import load_dotenv
import os

# bedrock chatbot 필요 라이브러리
import boto3

# langchain pdf 필요 라이브러리
# pdf reader , pdf splitter
from langchain_community.document_loaders import PyPDFLoader
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

# chromadb
from langchain_chroma import Chroma
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings

app = Flask(__name__)
CORS(app)

s3_bucket_name = os.getenv("AWS_BUCKET")
s3_region = os.getenv("AWS_REGION")
s3_access_key_id = os.getenv("S3_ACCESS_KEY")
s3_secret_access_key = os.getenv("S3_SECRET_ACCESS_KEY")
bedrock_region = os.getenv("BEDROCK_AWS_REGION")
bedrock_access_key_id = os.getenv("BEDROCK_ACCESS_KEY")
bedrock_secret_access_key = os.getenv("BEDROCK_SECRET_ACCESS_KEY")
host = os.getenv("BEDROCK_SECRET_ACCESS_KEY")

database_client = None  # db 위치
embedding = None  # 임베딩 방법
llm = None  # llm
retriever = None  # 검색기
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=bedrock_region,
    aws_access_key_id=bedrock_access_key_id,
    aws_secret_access_key=bedrock_secret_access_key,
)


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
            port=8000,
            ssl=False,
            headers=None,
            settings=Settings(allow_reset=True),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE,
        )
    elif loc == "ec2":
        database_client = chromadb.HttpClient(
            host="chroma",
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
        collection_name = f"{fileNum}_{fileName}"
        documents = PyPDFLoader(download_path).load_and_split()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        chroma_db = Chroma(
            client=database_client,
            collection_name=collection_name,
            embedding_function=embedding,
        )
        # print(f"{chroma_db._collection.count()}개 있음")
        if chroma_db._collection.count() == 0:
            # save to disk
            Chroma.from_documents(
                documents=docs,
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


@app.route("/question/langchain", methods=["POST"])
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
    retriever = chroma_db.as_retriever(search_kwargs={"k": 30})

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

    result = chain.invoke(userQuestion)
    return jsonify({"result": result.content})
    # def generate():
    #     # messages = [HumanMessage(content=userQuestion)]
    #     for chunk in chain.stream(userQuestion):
    #         # yield f"{chunk.content}\n"
    #         yield chunk.content
    #         # print(chunk.content, end="|", flush=True)
    # return Response(stream_with_context(generate()), content_type="text/event-stream")


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


# stream = response["body"]
# if stream:
#     for event in stream:
#         chunk = event["chunk"]
#         if chunk:
#             # print(json.loads(chunk.get("bytes").decode()))
#             result = json.loads(chunk["bytes"].decode())
#             print(result)
#             if result["type"] == "content_block_delta":
#                 yield result["delta"]["text"]


def check_file_exists_in_pdfs(filename):
    return os.path.isfile(f"./pdfs/{filename}")


@app.route("/test", methods=["GET"])
def testtest():
    print("테스트 데스와~")
    return jsonify({"result": "테스트 데스와~"})


# 챗봇 기본 세팅
setCahtBot()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
