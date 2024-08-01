# **연관 페이지 추천 기능을 가진 PDF 뷰어 Spread out**

## 프로젝트 개요

- 프로젝트 기간 :  2024.06.24 ~ 2024.07.27 (5주)
- 프로젝트 설명
    - pdf 파일 형식의 책을 업로드 하면 책을 분석하여 점과 연결의 그래프로 보여주고 챕터간의 관계 및 요약, 키워드를 보여줍니다.
    - 추가로 업로드한 pdf를 읽은 챗봇을 통해서 책의 내용에 대해서 질문하거나 요약하는 등 llm을 사용한 서비스를 제공합니다.
- 프로젝트 배경 및 의도
    - 전공서적, 인문 서적 등 읽기 어려운 책을 좀 더 빠르고 쉽게 읽을 수 있으면 좋겠다는 생각에서 시작한 프로젝트 입니다.
    - 추가로 챕터간 함께 보면 좋은 챕터를 연결하여 좀 더 책을 깊이있게 읽을 수 있도록 했습니다.
    - 책에서 나오는 어려운 단어나 내용을 좀 더 쉽게 설명해줄 수 있는 도우미가 있었으면 좋겠다고 생각했습니다.
- 프로젝트 아키텍처

![image](https://github.com/user-attachments/assets/f8484c24-5a5c-4fe7-a44c-b5b66aa2cd53)

## 기술스택

- 백엔드 : nodeJS, expressJS, MySQL, JWT,
- 프론트엔드 : NextJS, D3JS, reactPDF, JWT, axios, TypeScript,
- AI : Python, MySQL
- 챗봇 : Python, AWS Bedrock, Flask, Chroma, Langchian,
- 기타 : github runner, docker , AWS EC2, AWS S3, Postman

## 프로젝트 진행 단계
- 챗봇
    - Python Flask 기반으로 AWS Bedrock과 Langchian을 사용하여 LLM 챗봇 구축
    - PDF 텍스트를 저장하기 위한 텍스트 임베딩 기능 구현
    - 임베딩된 텍스트 데이터를 저장할 벡터 데이터베이스(Chroma DB) 구축
    - PDF를 챕터별로 분할 하여 요약, 키워드 추출, 챕터간 관계 설명 데이터 추출 및 저장 기능 구현
    - 챗봇 사용시 대화기록 및 업로드한 PDF 파일을 검색하여 필요한 데이터를 기반으로 대답할 수 있게 수정


## 프로젝트 세부 과정
### 챗봇

![image](https://github.com/user-attachments/assets/177d5703-773c-4784-8a11-05f6c0254999)

- Python Flask 기반으로 AWS Bedrock과 Langchian을 사용하여 LLM 챗봇 구축
    - 각 기술의 선택 사유
        1. python : 주 언어인 nodejs로 구현을 했을 때 레퍼런스가 적었고 레퍼런스가 실행되지 않은 경우가 많아서 python으로 다시 바꿧거 진행함
        2. Flask : django 보다 가볍고 빠르게 서버 기능을 구현 가능하고 서버 구조가 nodejs와 비슷해서 빠르게 진행 가능하다고 판단함
        3. AWS Bedrock : openAI의 chatGPT와 고민했지만 크래프토 정글에서 크래딧을 지원해줘서 AWS 를 사용하기로 했고 다양한 언어 모델을 쉽게 골라서 선택할 수 있었음
        4. Langchain : LLM들을 체인방식으로 연결할 수 있고 대화기록과 벡터데이터 베이스를 사용하여 검색하여 질문을 하기 쉽게 만들어져있음
- 임베딩된 텍스트 데이터를 저장할 벡터 데이터베이스(Chroma DB) 구축
    - docker를 사용해서 컨테이너 형식으로 인스턴스안에 구축함
    
    ```jsx
    // 구축 커맨드
    docker pull chromadb/chroma 
    docker run -d --name chroma -v /home/ubuntu/db/chroma:/chroma/chroma -p 7000:8000 -e ALLOW_RESET=TRUE chromadb/chroma 
    ```
    
    - Chroma DB 를 선택한 이유 :
        1. 임베딩 데이터를 저장하기 위해선 벡터 데이터베이스가 필요했음
        2. 유명한 엘라스틱 서치도 고민했지만 엘라스틱 서치를 사용하기 위해선 필요한 자원(메모리, 용량 등)이 많이 필요했고 인스턴스를 많이 업그레이드 했어야함
        3. 그래서 경량 모델이면서 docker로 레퍼런스가 잘 만들어져있는 chroma DB를 선택함
- PDF 텍스트를 저장하기 위한 텍스트 임베딩 기능 구현
    
    ![image](https://github.com/user-attachments/assets/02daf5af-e15d-4d06-9b6c-8944c2d25203)

    - LLM을 사용하여 서비스를 구현할때 PDF파일을 chatGPT 같이 읽게 하는건 불가능함
    - 이를 해결하려면 텍스트 데이터를 숫자데이터로 임베딩 해서 저장하고 사용해야함
    - 하지만 한번에 임베딩 가능한 텍스트의 양은 정해져있음 (약 8000 토큰)
    - 이를 해결하기 위해서 pdf 를 챕터 단위로 나눈 후 다시 임베딩 가능한 토큰 범위 안까지 요약 후 bedrock의 임베딩 AI(amazon titan embeding v1)을 사용해 임베딩과정 진행
- 챗봇 사용시 대화기록 및 업로드한 PDF 파일을 검색하여 필요한 데이터를 기반으로 대답할 수 있게 수정
    - 대화기록을 저장하기 위해서 LangchAin 에서 제공하는 InMemoryChatMessageHistory 클래스를 사용해서 대화 기록을 저장함
    - 이 저장된 대화는 대화방의 ID를 기준으로 누적되고 대화시 대화기록 프롬프트로 전달됨
    - 업로드한 PDF를 참조해서 대답하기 위해서 retriever(검색기) 를 사용해서 질문을 임베딩 데이터로 변환하여 의미적으로 가장 유사한 챕터를 찾아서 해당 내용들을 질문시 프롬프트로 함께 전달함
    - 즉 한번 질문에 ‘시스템 프롬프트 + 대화기록 + 질문한 내용과 관련된 검색 결과 + 질문‘ 이렇게 질문이 전달됨

## 프로젝트 결과

- 링크 : http://jungledaltong.shop/login
- git : https://github.com/WA360/SpreadoutBack/
- 영상 : https://youtu.be/3ZoiSgs1TYE

## 프로젝트 회고

- 한계점 및 개선 방안
    1.
