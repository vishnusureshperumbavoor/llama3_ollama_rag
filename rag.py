from typing import List 
import PyPDF2
from io import BytesIO
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma 
from langchain.chains import ConversationChain
from langchain.docstore.document import Document 
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama 
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

import chainlit as cl

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

@cl.on_chat_start
async def on_chat_start():
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content = "Please upload a PDF file to begin",
            accept = ["application/pdf"],
            max_size_mb = 20,
            timeout = 180,
        ).send()
    file = files[0]
    print(file)

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # read the pdf file
    pdf = PyPDF2.PdfReader(file.path)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()
        

    # split the text into chunks
    texts = text_splitter.split_text(pdf_text)

    # create a metadata for each chunks
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # create a chroma vector store
    embeddings = OllamaEmbeddings(model="mistral")
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas = metadatas
    )

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key = "chat_history",
        output_key = "answer",
        chat_memory = message_history,
        return_messages = True
    )

    # create a chain that uses the chroma vector store
    chain = ConversationChain.from_llm(
        ChatOllama(model="mistral"),
        chain_type = "stuff",
        retrieval = docsearch.as_retriever(),
        memory = memory,
        return_source_documents = True,
    )

    # let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]

    text_elements = []
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )

        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()