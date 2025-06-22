import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
from fpdf import FPDF
import re

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="YouTube Video Summarizer",
    layout="wide",
)

# -----------------------------
# TITLE
# -----------------------------
st.title("🎬 YouTube Video Summarizer")
st.write("Fetch transcript, clean it thoroughly, summarize with double-pass, highlight keywords, and download it!")

# -----------------------------
# FUNCTIONS
# -----------------------------

@st.cache_data(show_spinner=True)
def get_transcript(video_id):
    """Fetch and join transcript text safely."""
    try:
        raw_transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry['text'] for entry in raw_transcript])
        return text
    except Exception as e:
        return f"❌ Error fetching transcript: {str(e)}"

def clean_text(text):
    """Advanced cleaning with minimal content loss."""
    text = re.sub(r'\s+', ' ', text)  # normalize whitespace
    text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)  # fix missing space after punctuation
    text = re.sub(r'\[.*?\]', '', text)  # remove timestamps or bracket notes
    text = text.strip()
    return text

@st.cache_resource(show_spinner=True)
def get_summarizer():
    """Load summarizer model once."""
    return pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, chunk=500):
    """Double-pass summarization for high accuracy, chunked to avoid token limits."""
    summarizer = get_summarizer()
    summaries = []
    for i in range(0, len(text), chunk):
        part = text[i:i+chunk]
        summary = summarizer(part, max_length=120, min_length=30, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    # Second pass: summarize all summaries
    combined = " ".join(summaries)
    final_summary = summarizer(combined, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
    return final_summary

def highlight_keywords(text, keywords):
    """Add keyword highlights."""
    for kw in keywords:
        text = re.sub(f"(?i)({kw})", r"**\1**", text)
    return text

def make_pdf(summary_text):
    """Generate PDF with FPDF."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, summary_text)
    return pdf

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("🔗 Enter Video")
video_input = st.sidebar.text_input("YouTube URL or Video ID:", "")

# -----------------------------
# MAIN LOGIC
# -----------------------------
if video_input:
    # Extract ID if needed
    if "youtube.com" in video_input or "youtu.be" in video_input:
        try:
            video_id = re.findall(r"(?:v=|\/)([0-9A-Za-z_-]{11})", video_input)[0]
        except:
            st.error("Invalid YouTube link.")
            st.stop()
    else:
        video_id = video_input.strip()

    with st.spinner("Fetching transcript..."):
        transcript = get_transcript(video_id)

    if transcript.startswith("❌"):
        st.error(transcript)
    else:
        clean_transcript = clean_text(transcript)
        
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📝 Original Transcript")
            st.write(clean_transcript)

        with col2:
            st.subheader("✨ Generated Summary")
            with st.spinner("Summarizing... (double pass)"):
                summary = summarize_text(clean_transcript)

            # Highlight keywords (optional example: top 5 frequent words)
            words = re.findall(r'\w+', summary.lower())
            common = sorted(set(words), key=words.count, reverse=True)[:5]
            highlighted = highlight_keywords(summary, common)
            st.markdown(highlighted)

            st.success("✅ Summary Ready!")

            # Download buttons
            txt_filename = f"{video_id}_summary.txt"
            pdf_filename = f"{video_id}_summary.pdf"

            st.download_button(
                label="⬇️ Download as TXT",
                data=summary,
                file_name=txt_filename,
                mime="text/plain"
            )

            pdf = make_pdf(summary)
            pdf_output = bytes(pdf.output(dest='S').encode('latin1'))
            st.download_button(
                label="⬇️ Download as PDF",
                data=pdf_output,
                file_name=pdf_filename,
                mime="application/pdf"
            )

# -----------------------------
# FOOTER
# -----------------------------
st.info("Built with ❤️ using Streamlit, youtube-transcript-api, and transformers. Built in hopes of getting recruited. Built by CS24B2014 Anjana Chandru")
