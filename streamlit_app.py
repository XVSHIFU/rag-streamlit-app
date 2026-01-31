import streamlit as st
import os
import sys
from dotenv import load_dotenv

# LangChain 核心
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough

# 向量数据库和 Embeddings
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings

# 智谱大模型
from zhipuai_llm import ZhipuaiLLM

# 加载 .env (本地调试用)
load_dotenv()


# ----------- 自定义智谱 Embeddings 类 -----------
class ZhipuAIEmbeddings(Embeddings):
    def __init__(self):
        from zhipuai import ZhipuAI
        api_key = os.environ.get("ZHIPUAI_API_KEY")
        if not api_key:
            raise ValueError("未提供 ZHIPUAI_API_KEY，请在 Streamlit Secrets 中配置")
        self.client = ZhipuAI(api_key=api_key)

    def embed_documents(self, texts):
        return [self.client.embedding(text)["embedding"] for text in texts]

    def embed_query(self, text):
        return self.client.embedding(text)["embedding"]


# ----------- 获取向量检索器 -----------
def get_retriever(documents=None):
    embedding = ZhipuAIEmbeddings()

    # 如果没有传文档，使用示例文档
    if documents is None:
        documents = [
            {"page_content": "南瓜书是《机器学习》（西瓜书）的配套辅导书，用于帮助理解西瓜书的内容。"},
            {"page_content": "Prompt Engineering 是为大语言模型设计和优化提示的技术。"}
        ]

    vectordb = Chroma.from_documents(
        documents,
        embedding_function=embedding,
        persist_directory=None  # Cloud 上不写入磁盘
    )
    return vectordb.as_retriever(search_kwargs={"k": 3})


# ----------- 文档合并函数 -----------
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs["context"])


# ----------- 构建 QA 链 -----------
def get_qa_history_chain():
    retriever = get_retriever()
    llm = ZhipuaiLLM(model_name="glm-4-plus", temperature=0.1, api_key=os.environ.get("ZHIPUAI_API_KEY"))

    condense_question_prompt = ChatPromptTemplate([
        ("system", "请根据聊天记录总结用户最近的问题，如果没有多余聊天记录则返回用户的问题。"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])

    retrieve_docs = RunnableBranch(
        (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever),
        condense_question_prompt | llm | StrOutputParser() | retriever,
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是问答助手。使用检索到的上下文回答用户问题，不知道就说不知道，简洁回答。\n\n{context}"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])

    qa_chain = (
        RunnablePassthrough().assign(context=combine_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    qa_history_chain = RunnablePassthrough().assign(
        context=retrieve_docs
    ).assign(answer=qa_chain)

    return qa_history_chain


# ----------- 流式回答生成 -----------
def gen_response(chain, input, chat_history):
    response = chain.stream({
        "input": input,
        "chat_history": chat_history
    })
    for res in response:
        if "answer" in res:
            yield res["answer"]


# ----------- Streamlit 界面 -----------
def main():
    st.markdown("### 🦜🔗 RAG 大模型问答演示")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "qa_history_chain" not in st.session_state:
        st.session_state.qa_history_chain = get_qa_history_chain()

    messages_container = st.container(height=550)

    # 显示历史消息
    for role, msg in st.session_state.messages:
        with messages_container.chat_message(role):
            st.write(msg)

    if prompt := st.chat_input("请输入问题"):
        st.session_state.messages.append(("human", prompt))
        with messages_container.chat_message("human"):
            st.write(prompt)

        answer_stream = gen_response(
            chain=st.session_state.qa_history_chain,
            input=prompt,
            chat_history=st.session_state.messages
        )

        with messages_container.chat_message("ai"):
            output = st.write_stream(answer_stream)

        st.session_state.messages.append(("ai", output))


if __name__ == "__main__":
    main()
