import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
from fpdf import FPDF
import re

# ------------------------------------
# CONFIG
# ------------------------------------
st.set_page_config(page_title="📺 YouTube Summarizer", layout="wide")

st.title("🎥 YouTube Video Summarizer")
st.write("Fetch, clean, summarize with double pass, highlight key words, and download!")

# ------------------------------------
# FUNCTIONS
# ------------------------------------

@st.cache_data(show_spinner=True)
def fetch_transcript(video_id):
    """Fetch transcript from YouTube."""
    try:
        data = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([item['text'] for item in data])
    except Exception as e:
        return f"❌ Error: {e}"

def clean_transcript(text):
    """Clean without losing info."""
    text = re.sub(r'\[.*?\]', '', text)  # remove brackets
    text = re.sub(r'\s+', ' ', text)     # normalize spaces
    text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)  # fix missing space after punctuation
    return text.strip()

@st.cache_resource(show_spinner=True)
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

def chunk_text(text, max_length=800):
    """Chunk text for summarizer."""
    sentences = text.split('. ')
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) < max_length:
            current += s + ". "
        else:
            chunks.append(current.strip())
            current = s + ". "
    if current:
        chunks.append(current.strip())
    return chunks

def double_summarize(text):
    """Two-pass summary."""
    summarizer = load_summarizer()
    chunks = chunk_text(text)
    first_pass = [summarizer(c, max_length=120, min_length=30, do_sample=False)[0]['summary_text'] for c in chunks]
    combined = " ".join(first_pass)
    final_summary = summarizer(combined, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
    return final_summary

def highlight(summary, words):
    """Highlight words."""
    for w in words:
        summary = re.sub(f'(?i)({w})', r'**\1**', summary)
    return summary

def to_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    return pdf.output(dest='S').encode('latin-1')

# ------------------------------------
# SIDEBAR INPUT
# ------------------------------------

video_input = st.sidebar.text_input("🎥 Enter YouTube Video URL or ID:")

if video_input:
    if "youtu" in video_input:
        ids = re.findall(r"(?:v=|\/)([0-9A-Za-z_-]{11})", video_input)
        if not ids:
            st.error("❌ Could not extract video ID.")
            st.stop()
        video_id = ids[0]
    else:
        video_id = video_input.strip()

    with st.spinner("Fetching transcript..."):
        raw = fetch_transcript(video_id)

    if raw.startswith("❌"):
        st.error(raw)
    else:
        cleaned = clean_transcript(raw)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📜 Full Transcript")
            st.write(cleaned)

        with col2:
            st.subheader("✨ Double-Pass Summary")
            with st.spinner("Generating..."):
                summary = double_summarize(cleaned)
                words = re.findall(r'\w+', summary.lower())
                freq = sorted(set(words), key=words.count, reverse=True)[:5]
                highlighted = highlight(summary, freq)
                st.markdown(highlighted)

            st.download_button(
                "⬇️ Download TXT",
                data=summary,
                file_name=f"{video_id}_summary.txt",
                mime="text/plain"
            )

            pdf_data = to_pdf(summary)
            st.download_button(
                "⬇️ Download PDF",
                data=pdf_data,
                file_name=f"{video_id}_summary.pdf",
                mime="application/pdf"
            )
st.info("Built with ❤️ using Streamlit, youtube-transcript-api, and transformers. Built in hopes of getting recruited. Built by CS24B2014 Anjana Chandru")
