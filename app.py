import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient


#----------------------------------------------------- Streamlit basic config ---------------------------------------------

st.set_page_config(page_title="CelestIA - Zodiac Chat", page_icon="🔮")
st.title("🔮 CelestIA - Zodiac Chat")
st.write("Ask about zodiac signs, compatibility, or birth dates.")

#----------------------------------------------------- Getting local secrets key or by cloud set up ------------------------

# get secret key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        groq_api_key = None

#get qdrant url
qdrant_url = os.getenv("QDRANT_URL")
if not qdrant_url:
    try:
        qdrant_url = st.secrets("QDRANT_URL")
    except Exception:
        qdrant_url = None

#get qdrant secrets
qdrant_api_key = os.getenv("QDRANT_API_KEY")
if not qdrant_api_key:
    try:
        qdrant_api_key = st.secrets("QDRANT_API_KEY")
    except Exception:
        qdrant_api_key = None

# -------------------------------------------------Instance llm/embedding/vectorstore --------------------------------------------------------------


llm = ChatGroq(
    api_key=groq_api_key,
    temperature=0.6,
    model_name="llama-3.1-8b-instant"
)

#the embedding model must to be the same used in the data_ingestion
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

collection_name = "astrology-zodiac"

client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, prefer_grpc=False)
vectorstores = Qdrant(
client=client,
collection_name=collection_name,
embeddings=embedding,
)

# vectorstores = Qdrant(
#     url = qdrant_url,
#     api_key = qdrant_api_key,
#     embedding = embedding,
#     collection_name = collection_name,
#     prefer_grpc=False,  # keep HTTP to avoid some cloud issues
# )
retriever = vectorstores.as_retriever(search_kwargs = {"k":3})

# ----------------------------------------------------Defining prompt---------------------------------------------------------------------

RAG_PROMPT = ChatPromptTemplate.from_template(
    """
You are a friendly astrology assistant. You answer using the same language that the user talk you you.
You MUST use ONLY the information provided in the context below.
If the answer is not in the context, say that it is not in your material.
If the user asks for future prediction, say you do not predict the future.

Context:
{context}

User question:
{question}

Answer (short, clear, friendly):
"""
)

output_parser = StrOutputParser()

# ------------------------------------------------------- Main function -------------------------------------------------------------------

def answer_with_rag(user_question:str) -> str:
    """
        Retrieves relevant astrology chunks from Qdrant
        and asks the Groq LLM to answer using only that context.
    """

    docs = retriever.invoke(user_question)
    context = "\n\n".join(doc.page_content for doc in docs)

    #build final prompt
    final_prompt = RAG_PROMPT.format(context=context,question=user_question)

    llm_response = llm.invoke(final_prompt)

    return output_parser.parse(llm_response),context

# --------------------------------------------------------------------------------------------------------------------------------------------



# history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi there! Tell me a sign or a birth date, and I'll explain! 👀"}
    ]

# show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# input box
user_input = st.chat_input("Type your astrology question here...")

if user_input:
    # show user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # get RAG answer
    answer,context = answer_with_rag(user_input)

    # show assistant message
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer.content)
        # st.markdown("context:  " + context)


