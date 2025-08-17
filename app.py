# -------------------------------------------------
# app.py – Streamlit demo (CPU‑only)
# -------------------------------------------------
import os
import torch
import streamlit as st

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ------------------------------------------------------------------
# 1️⃣  USER SETTINGS – change these to try a different model ----------
# ------------------------------------------------------------------
REPO_ID = "EleutherAI/gpt-neo-1.3B"   # ← replace with any model from the list
# QUANT can be: None, "4bit", "8bit"
QUANT = "4bit"                       # 4‑bit is the sweet spot for >1 B params
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# 2️⃣  Helper: load the model once and cache it with @st.cache_resource
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_generator():
    """
    Load the model + tokenizer and wrap them in a HF `pipeline`.
    The function is cached, so it runs only the first time the app starts.
    """
    # ---- tokenizer -------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID, use_fast=True)

    # ---- model loading options ------------------------------------
    model_kwargs = {
        "torch_dtype": torch.float16,   # fp16 works on most modern CPUs
        "device_map": "cpu",
    }

    if QUANT == "4bit":
        model_kwargs.update(
            {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
            }
        )
    elif QUANT == "8bit":
        model_kwargs["load_in_8bit"] = True

    # ---- actual model ---------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(REPO_ID, **model_kwargs)

    # ---- pipeline -------------------------------------------------
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
    )
    return generator


# ------------------------------------------------------------------
# 3️⃣  Initialise Streamlit UI
# ------------------------------------------------------------------
st.set_page_config(page_title="CPU‑only LLM Chat", page_icon="🤗", layout="centered")
st.title("🤗 CPU‑only LLM Chat (Streamlit)")

# Show which model we are using
model_desc = f"**Model:** `{REPO_ID}`  |  **Quantisation:** `{QUANT or 'fp16'}`"
st.caption(model_desc)

# ------------------------------------------------------------------
# 4️⃣  Session state – keep the conversation across reruns
# ------------------------------------------------------------------
if "history" not in st.session_state:
    # history is a list of (user, assistant) tuples
    st.session_state.history = []

# ------------------------------------------------------------------
# 5️⃣  UI – input box + chat display
# ------------------------------------------------------------------
# Use the optional `streamlit_chat` component if it is installed.
# If you don’t have it, the fallback is plain markdown.
try:
    from streamlit_chat import message as chat_message
    use_chat_component = True
except Exception:
    use_chat_component = False

# Render the existing chat history (most recent at the bottom)
for user_msg, bot_msg in st.session_state.history:
    if use_chat_component:
        chat_message(user_msg, is_user=True, key=f"user_{hash(user_msg)}")
        chat_message(bot_msg, is_user=False, key=f"bot_{hash(bot_msg)}")
    else:
        st.markdown(f"**You:** {user_msg}")
        st.markdown(f"**Assistant:** {bot_msg}")

# ------------------------------------------------------------------
# 6️⃣  Text input – the user writes a new message
# ------------------------------------------------------------------
user_input = st.chat_input("Ask something …") if hasattr(st, "chat_input") else st.text_input(
    "Your message:", key="user_input"
)

if user_input:
    # -------------------------------------------------------------
    # a) Append the user message to the history (so it shows immediately)
    # -------------------------------------------------------------
    st.session_state.history.append((user_input, ""))  # placeholder for the answer

    # -------------------------------------------------------------
    # b) Build a single prompt that contains the whole conversation.
    # -------------------------------------------------------------
    prompt = ""
    for u, b in st.session_state.history:
        # The placeholder "" for the bot answer will be ignored on the next turn
        if b == "":
            prompt += f"User: {u}\nAssistant:"
        else:
            prompt += f"User: {u}\nAssistant: {b}\n"

    # -------------------------------------------------------------
    # c) Run generation (this may take a few seconds)
    # -------------------------------------------------------------
    generator = load_generator()
    generated = generator(prompt)[0]["generated_text"]

    # -------------------------------------------------------------
    # d) Extract only the newly generated assistant text.
    # -------------------------------------------------------------
    # The model returns the full prompt + answer; we slice out the answer.
    answer = generated.split("Assistant:")[-1].strip()
    # In case the model repeats the "User:" token, cut it off.
    answer = answer.split("\nUser:")[0].strip()

    # -------------------------------------------------------------
    # e) Replace the placeholder with the real answer
    # -------------------------------------------------------------
    st.session_state.history[-1] = (user_input, answer)

    # -------------------------------------------------------------
    # f) Rerun the script so the new message appears in the UI
    # -------------------------------------------------------------
    st.experimental_rerun()
