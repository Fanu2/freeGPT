# -------------------------------------------------
# app.py – Streamlit demo (CPU-only, safe fallback)
# -------------------------------------------------
import os
import torch
import streamlit as st

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ------------------------------------------------------------------
# 1️⃣  USER SETTINGS – change these to try a different model ----------
# ------------------------------------------------------------------
REPO_ID = "EleutherAI/gpt-neo-1.3B"
QUANT = "4bit"          # "4bit", "8bit" or None
# ------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_generator():
    """
    Load the model + tokenizer with optional quantization.
    If 4-bit load fails, we fall back to fp16.
    """
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID, use_fast=True)

    # Initial arguments
    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "cpu"
    }

    # Try quantization
    if QUANT in ("4bit", "8bit"):
        try:
            if QUANT == "4bit":
                model_kwargs.update({
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": torch.float16,
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_quant_type": "nf4",
                })
            elif QUANT == "8bit":
                model_kwargs["load_in_8bit"] = True

            # Try loading with quant
            model = AutoModelForCausalLM.from_pretrained(REPO_ID, **model_kwargs)

        except Exception as e:
            # Fallback if quantization fails
            st.warning(
                f"⚠️ Quantized load failed ({e}). Falling back to fp16 / CPU."
            )
            # Remove quant flags
            model_kwargs.pop("load_in_4bit", None)
            model_kwargs.pop("load_in_8bit", None)
            model = AutoModelForCausalLM.from_pretrained(REPO_ID, torch_dtype=torch.float16, device_map="cpu")

    else:
        model = AutoModelForCausalLM.from_pretrained(REPO_ID, **model_kwargs)

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
# UI + Chat + Session logic remains the same from here on
# -------------------------------------------------

st.set_page_config(page_title="CPU-only LLM Chat", page_icon="🤗", layout="centered")
st.title("🤗 CPU-only LLM Chat (Streamlit)")

model_desc = f"**Model:** `{REPO_ID}`  |  **Quantisation:** `{QUANT or 'fp16'}`"
st.caption(model_desc)

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
