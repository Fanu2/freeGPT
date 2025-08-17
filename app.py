# -------------------------------------------------
# app.py – Streamlit CPU demo (no quant, fast model)
# -------------------------------------------------
import os
import torch
import streamlit as st

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ------------------------------------------------------------------
# 1️⃣  USER SETTINGS – no quant, small model, CPU-safe
# ------------------------------------------------------------------
REPO_ID = "distilgpt2"   # lighter model, no HF token needed
QUANT = None             # No quantization in this version on CPU
# ------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_generator():
    """
    Load the model + tokenizer on CPU. No quantization to avoid runtime errors.
    """
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        REPO_ID,
        torch_dtype=torch.float32,
        device_map="cpu"
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
    )
    return generator


# -------------------------------------------------
# Streamlit UI + Session state
# -------------------------------------------------
st.set_page_config(page_title="CPU LLM Chat", page_icon="🤖", layout="centered")
st.title("💬 Lightweight CPU LLM Chat (Streamlit)")

st.caption(f"**Model:** `{REPO_ID}`  |  **Quantisation:** `None`")

if "history" not in st.session_state:
    st.session_state.history = []

try:
    from streamlit_chat import message as chat_message
    use_chat_component = True
except Exception:
    use_chat_component = False

for user_msg, bot_msg in st.session_state.history:
    if use_chat_component:
        chat_message(user_msg, is_user=True, key=f"user_{hash(user_msg)}")
        chat_message(bot_msg, is_user=False, key=f"bot_{hash(bot_msg)}")
    else:
        st.markdown(f"**You:** {user_msg}")
        st.markdown(f"**Assistant:** {bot_msg}")

user_input = st.chat_input("Ask something …") if hasattr(st, "chat_input") else st.text_input(
    "Your message:", key="user_input"
)

if user_input:
    st.session_state.history.append((user_input, ""))

    prompt = ""
    for u, b in st.session_state.history:
        if b == "":
            prompt += f"User: {u}\nAssistant:"
        else:
            prompt += f"User: {u}\nAssistant: {b}\n"

    generator = load_generator()
    generated = generator(prompt)[0]["generated_text"]

    answer = generated.split("Assistant:")[-1].strip()
    answer = answer.split("\nUser:")[0].strip()

    st.session_state.history[-1] = (user_input, answer)
    st.experimental_rerun()
