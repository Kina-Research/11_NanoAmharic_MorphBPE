"""
streamlit_app.py — Interactive Streamlit Dashboard for NanoAmharic-MorphBPE.
Features:
- Amharic Storytelling & Prompt Completion
- Interactive Multi-Turn Chat
- MorphBPE vs Standard BPE Morpheme Decomposition Visualizer
- Empirical Benchmark & Quantization Metrics
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import streamlit as st
except ImportError:
    print("Streamlit is not installed. To run this app, execute: pip install streamlit")
    sys.exit(0)

from src.app.inference_engine import NanoAmharicInferenceEngine
from src.tokenizer.morphbpe import MorphBPETokenizer
from src.tokenizer.morph_rules import identify_morphological_boundaries, crosses_morphological_boundary

st.set_page_config(
    page_title="NanoAmharic-MorphBPE Studio",
    page_icon="🇪🇹",
    layout="wide",
)

st.title("🇪🇹 ናኖ-አማርኛ (NanoAmharic-MorphBPE)")
st.caption("Morphology-Constrained Subword Tokenization & CPU-Trainable Decoder-Only Language Model (5.75M Parameters)")

# Sidebar Settings
st.sidebar.header("⚙️ የሞዴል ቅንብሮች (Settings)")
use_int8 = st.sidebar.checkbox("INT8 Quantization (9.44 MB)", value=False)
temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=1.5, value=0.7, step=0.05)
max_tokens = st.sidebar.slider("Max New Tokens", min_value=10, max_value=120, value=50, step=5)
top_k = st.sidebar.slider("Top-K", min_value=5, max_value=100, value=40, step=5)
top_p = st.sidebar.slider("Top-P", min_value=0.1, max_value=1.0, value=0.9, step=0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("**Hardware Constraint:** Single Workstation Multicore CPU")
st.sidebar.markdown("**Context Window:** 128 tokens")
st.sidebar.markdown("**Vocabulary:** 4,000 subwords")


@st.cache_resource
def load_engine(is_int8: bool):
    return NanoAmharicInferenceEngine(use_int8=is_int8)


@st.cache_resource
def load_tokenizers():
    morph_dir = PROJECT_ROOT / "eval" / "results" / "tokenizer" / "MorphBPE_v4k_l050"
    std_dir = PROJECT_ROOT / "eval" / "results" / "tokenizer" / "Standard_BPE_v4k"
    morph_tok = MorphBPETokenizer.load(morph_dir)
    std_tok = MorphBPETokenizer.load(std_dir)
    return morph_tok, std_tok


engine = load_engine(use_int8)
morph_tok, std_tok = load_tokenizers()

tabs = st.tabs(["📖 ተረትና ታሪክ (Story)", "💬 ውይይት (Chat)", "✍️ ጽሑፍ ማጠናቀቂያ (Completion)", "🔍 Tokenizer Lab", "📊 የምርምር ውጤቶች (Benchmarks)"])

# 1. Story Tab
with tabs[0]:
    st.subheader("የአማርኛ ተረትና ታሪክ ማመንጫ")
    topic = st.text_input("የታሪኩ ርዕስ ወይም መነሻ ሐሳብ፡", "በጥንት ዘመን አንድ ጥበበኛ ንጉሥ")
    if st.button("ታሪክ ፍጠር", key="btn_story"):
        with st.spinner("ታሪክ እየተፈጠረ ነው..."):
            res = engine.generate_story(topic=topic, max_tokens=max_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
            st.text_area("የተፈጠረው ታሪክ፡", res["full_text"], height=200)
            st.info(f"⚡ ፍጥነት፡ {res['tokens_per_second']} tok/s | ⏱ ጊዜ፡ {res['elapsed_seconds']}s | 📝 ቃላት፡ {res['tokens_generated']} tokens")

# 2. Chat Tab
with tabs[1]:
    st.subheader("በአማርኛ ይወያዩ")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "ሰላም! እኔ ናኖ-አማርኛ ነኝ። ዛሬ በምን ልርዳዎ?"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if user_prompt := st.chat_input("ጥያቄዎን ወይም መልእክትዎን እዚህ ይጻፉ..."):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.write(user_prompt)

        with st.chat_message("assistant"):
            chat_history = [{"user": m["content"], "assistant": ""} for m in st.session_state.messages if m["role"] == "user"]
            res = engine.chat(user_prompt, conversation_history=chat_history, max_tokens=max_tokens, temperature=temperature)
            reply = res["assistant_response"]
            st.write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

# 3. Completion Tab
with tabs[2]:
    st.subheader("ጽሑፍ ማጠናቀቂያ")
    prompt = st.text_area("የመነሻ ጽሑፍ (Prompt)፡", "የኢትዮጵያ ታሪካዊ ከተሞች")
    if st.button("አጠናቅቅ", key="btn_complete"):
        with st.spinner("ጽሑፉ እየተጠናቀቀ ነው..."):
            res = engine.generate(prompt=prompt, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
            st.text_area("ውጤት፡", res["full_text"], height=160)
            st.info(f"⚡ ፍጥነት፡ {res['tokens_per_second']} tok/s | ⏱ ጊዜ፡ {res['elapsed_seconds']}s | 📝 ቃላት፡ {res['tokens_generated']} tokens")

# 4. Tokenizer Lab Tab
with tabs[3]:
    st.subheader("MorphBPE vs Standard BPE Tokenizer Lab")
    tok_input = st.text_input("የሚተነተን ቃል ወይም ዐረፍተ-ነገር፡", "በመንግሥታችን አልተረዱልንም ኢትዮጵያዊነት")
    if st.button("በሞርፎሎጂ ተንትን", key="btn_tok"):
        words = tok_input.split()
        st.markdown(f"**MorphBPE Subwords:** `{morph_tok.tokenize(tok_input)}`")
        st.markdown(f"**Standard BPE Subwords:** `{std_tok.tokenize(tok_input)}`")
        st.markdown("---")
        for w in words:
            b = sorted(list(identify_morphological_boundaries(w)))
            m_tokens = morph_tok.tokenize(w)
            s_tokens = std_tok.tokenize(w)
            st.markdown(f"**ቃል:** `{w}` | **የተለዩ ድንበሮች:** `{b}`")
            st.markdown(f"- MorphBPE: `{m_tokens}`")
            st.markdown(f"- Standard BPE: `{s_tokens}`")

# 5. Benchmarks Tab
with tabs[4]:
    st.subheader("የምርምርና የሙከራ ውጤቶች (Empirical Benchmark Summary)")
    st.markdown("""
    | መለኪያ (Metric) | Standard BPE | MorphBPE ($\lambda=0.5$) | ልዩነት ($\Delta$) | ስታቲስቲክስ |
    |---|---|---|---|---|
    | **Validation Loss** | 3.5533 | **3.5118** | -0.0415 | የላቀ |
    | **Word Perplexity** | 92.84 | **89.25** | -3.59 | የላቀ |
    | **Bits-per-Character (BPC)** | 1.2776 | **1.2664** | -0.0112 | $p < 0.001$ |
    | **የሞርፎሎጂ ድንበር ጥበቃ** | 17.89% | **25.10%** | +40.3% rel | $p < 0.001$ |
    | **የሞርፎሎጂ ስህተት ምጣኔ** | 3.03% | **1.46%** | -51.8% rel | $p < 0.001$ |
    | **የስርወ-ቃል (Stem) ጥበቃ** | 94.96% | **97.54%** | +2.58% abs | $p < 0.001$ |
    """)
    st.markdown("**Dynamic INT8 Quantization:** 65.82 MB $\\rightarrow$ 9.44 MB (6.97× ቅነሳ) | RAM Footprint: ~120 MB | CPU Throughput: 51.6 tok/s")
