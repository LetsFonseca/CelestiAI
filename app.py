import streamlit as st
from groq import Groq
import os

st.set_page_config(page_title="Chat dos Signos (Groq)", page_icon="🔮")

# pega chave dos secrets
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    try:
        # st.secrets pode levantar erro se o arquivo não existir
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        groq_api_key = None

client = Groq(api_key=groq_api_key)

ZODIAC_CONTEXT = """
Você é um assistente de astrologia, simpático, direto e em português do Brasil.
Você fala sobre os 12 signos do zodíaco: Áries, Touro, Gêmeos, Câncer, Leão, Virgem,
Libra, Escorpião, Sagitário, Capricórnio, Aquário e Peixes.

Regras:
- Se o usuário der uma data (ex: 15/08), tente dizer o signo.
- Se ele disser só o signo, descreva: elemento, qualidades e alertas.
- Se perguntar compatibilidade, explique rapidamente fogo/ar e terra/água e depois fale do par.
- Se perguntar “mapa”, diga que precisa de data, hora e cidade.
- Não prometa previsão do futuro. Mantenha o tom leve.

Se fugir do tema, puxe de volta para astrologia.
"""

st.title("🔮 Chat dos Signos (IA via Groq)")
st.write("Pergunte sobre signos, compatibilidade ou datas de nascimento.")

# histórico
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Oi! Me diz um signo ou uma data que eu te conto 👀"}
    ]

# mostra histórico
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def call_groq(user_msg: str) -> str:
    if groq_api_key is None:
        return "⚠️ Falta a variável GROQ_API_KEY nos secrets."

    # monta mensagens: system + histórico + nova pergunta
    messages = [
        {"role": "system", "content": ZODIAC_CONTEXT},
    ]

    for m in st.session_state.messages:
        messages.append({"role": m["role"], "content": m["content"]})

    messages.append({"role": "user", "content": user_msg})

    # modelo do Groq — pode trocar por "llama-3.1-70b-versatile" se quiser mais forte
    chat = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.7,
        max_tokens=400,
    )

    return chat.choices[0].message.content

# entrada do usuário
if prompt := st.chat_input("Digite sua pergunta sobre signos..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    answer = call_groq(prompt)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
