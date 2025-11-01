import streamlit as st
from groq import Groq
import os

st.set_page_config(page_title="CelestIA - Zodiac Chat", page_icon="🔮")

# get secret key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        groq_api_key = None

client = Groq(api_key=groq_api_key)

ZODIAC_CONTEXT = """
You are an astrology assistant — friendly, straightforward, and you speak the language that the user speak with you.
You talk about the 12 zodiac signs: Aries, Taurus, Gemini, Cancer, Leo, Virgo, Libra, Scorpio, Sagittarius, Capricorn, Aquarius, and Pisces.

Rules:

* If the user provides a date (e.g., 15/08), try to determine the zodiac sign.
* If the user only mentions a sign, describe its element, strengths, and cautions.
* If the user asks about compatibility, briefly explain the fire/air and earth/water dynamics, then comment on the specific pair.
* If the user asks about a “birth chart,” explain that you need the date, time, and city of birth.
* Do not promise to predict the future. Keep the tone light and friendly.
* If the user strays off topic, steer the conversation back to astrology.
"""

st.title("🔮 CelestIA - Zodiac Chat")
st.write("Ask about zodiac signs, compatibility, or birth dates.")

# history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi there! Tell me a sign or a birth date, and I'll explain! 👀"}
    ]

# show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def call_groq(user_msg: str) -> str:
    if groq_api_key is None:
        return "⚠️ Missing GROQ_API_KEY secrets key."

    # build the message
    messages = [
        {"role": "system", "content": ZODIAC_CONTEXT},
    ]

    for m in st.session_state.messages:
        messages.append({"role": m["role"], "content": m["content"]})

    messages.append({"role": "user", "content": user_msg})

    
    chat = client.chat.completions.create(
        # model="llama-3.1-8b-instant",
        model = "llama-3.1-70b-versatile",
        messages=messages,
        temperature=0.7,
        max_tokens=400,
    )

    return chat.choices[0].message.content

# User input
if prompt := st.chat_input("Type your question about zodiac signs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    answer = call_groq(prompt)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
