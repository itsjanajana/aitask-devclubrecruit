import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
from fpdf import FPDF
import re
import nltk
from nltk.tokenize import sent_tokenize
from difflib import SequenceMatcher
from collections import Counter
import string

nltk.download('punkt')

def thorough_clean(text):
    text = re.sub(r'\[\d{1,2}:\d{2}(:\d{2})?\]', '', text)
    text = re.sub(r'\bSpeaker \d+:\s*', '', text)
    text = re.sub(r'\[(music|applause|laughter|noise|cough)\]', '', text, flags=re.I)
    text = re.sub(r'-\s*\n\s*', '', text)
    filler_pattern = r'\b(um|uh|like|you know|you see|so|actually|basically|right|okay|well)\b'
    text = re.sub(filler_pattern, '', text, flags=re.I)
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_text(text, max_tokens=600, overlap_tokens=200):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0
    for sentence in sentences:
        sent_len = len(sentence.split())
        if current_len + sent_len > max_tokens:
            chunks.append(" ".join(current_chunk))
            overlap_count = 0
            overlap_sentences = []
            for s in reversed(current_chunk):
                overlap_count += len(s.split())
                overlap_sentences.insert(0, s)
                if overlap_count >= overlap_tokens:
                    break
            current_chunk = overlap_sentences
            current_len = sum(len(s.split()) for s in current_chunk)
        current_chunk.append(sentence)
        current_len += sent_len
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def generate_summary(model, text, max_len, min_len):
    return model(text, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def create_pdf(summary, video_id):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "YouTube Video Summary", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    for line in summary.split('. '):
        pdf.multi_cell(0, 10, line.strip() + '.')
    filename = f"summary_{video_id}.pdf"
    pdf.output(filename)
    return filename

def extract_keywords(text, num_keywords=10):
    text_clean = text.lower().translate(str.maketrans('', '', string.punctuation))
    words = text_clean.split()
    stopwords = set([
        "the", "and", "to", "of", "a", "in", "is", "that", "it", "on", "for", "with",
        "as", "this", "are", "was", "but", "be", "at", "by", "or", "an", "if", "from",
        "so", "we", "can", "will", "not", "they", "all", "has", "have"
    ])
    filtered = [w for w in words if w not in stopwords and len(w) > 2]
    counter = Counter(filtered)
    keywords = [word for word, count in counter.most_common(num_keywords)]
    return keywords

def highlight_keywords(summary, keywords):
    for kw in keywords:
        summary = re.sub(r'\b(' + re.escape(kw) + r')\b', f"**{kw.upper()}**", summary, flags=re.I)
    return summary

@st.cache_resource(show_spinner=False)
def load_model(name):
    return pipeline("summarization", model=name)

def main():
    st.title("🎥 YouTube Video Summarizer — Rigorous & Accurate")

    video_url = st.text_input("Paste YouTube video URL")
    model_name = st.selectbox("Choose summarization model", ["facebook/bart-large-cnn", "google/pegasus-xsum"])

    summary_len = st.radio("Summary length", ["Detailed", "Medium", "Short"])

    length_map = {
        "Detailed": (300, 150),
        "Medium": (130, 50),
        "Short": (60, 20)
    }

    if video_url:
        try:
            video_id = video_url.split("v=")[-1].split("&")[0]
            st.info(f"Fetching transcript for video ID: {video_id}...")
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = transcript_list.find_transcript(['en'])
            raw_text = " ".join([x['text'] for x in transcript.fetch()])

            st.expander("Show Raw Transcript").write(raw_text)

            cleaned_text = thorough_clean(raw_text)
            st.expander("Show Cleaned Transcript").write(cleaned_text)

            if st.button("Generate Summary"):
                with st.spinner("Generating summary..."):
                    model = load_model(model_name)
                    max_len, min_len = length_map[summary_len]

                    chunks = chunk_text(cleaned_text, max_tokens=600, overlap_tokens=200)
                    chunk_summaries = []
                    progress = st.progress(0)
                    for i, chunk in enumerate(chunks):
                        chunk_sum = generate_summary(model, chunk, max_len, min_len)
                        chunk_summaries.append(chunk_sum)
                        progress.progress((i+1)/len(chunks))

                    combined_summary = " ".join(chunk_summaries)

                    summary_1 = generate_summary(model, combined_summary, max_len*2, min_len*2)
                    summary_2 = generate_summary(model, combined_summary, max_len, max(20, min_len//2))

                    sim_score = similarity(summary_1, summary_2)
                    st.write(f"Similarity between summaries: {sim_score:.2f}")

                    if sim_score < 0.85:
                        final_summary = summary_1 if len(summary_1) > len(summary_2) else summary_2
                        st.warning("Low similarity between summary passes; returning longer summary to preserve content.")
                    else:
                        final_summary = summary_2
                        st.success("Summaries consistent, returning concise summary.")

                    keywords = extract_keywords(final_summary)
                    highlighted_summary = highlight_keywords(final_summary, keywords)

                    st.markdown("### Final Summary with Keywords Highlighted")
                    st.markdown(highlighted_summary)

                    st.markdown("### Side-by-Side Transcript and Summary")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.header("Cleaned Transcript")
                        st.write(cleaned_text)
                    with col2:
                        st.header("Summary")
                        st.write(final_summary)

                    pdf_file = create_pdf(final_summary, video_id)
                    with open(pdf_file, "rb") as f:
                        st.download_button("Download PDF 📄", f, file_name=pdf_file, mime="application/pdf")

                    txt_file = f"summary_{video_id}.txt"
                    with open(txt_file, "w") as f:
                        f.write(final_summary)
                    with open(txt_file, "rb") as f:
                        st.download_button("Download TXT 📝", f, file_name=txt_file, mime="text/plain")

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
