import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import re

st.title("Simple YouTube Video Summarizer")

video_input = st.text_input("Enter YouTube video URL or ID:")

def get_video_id(url_or_id):
    if "youtu" in url_or_id:
        result = re.findall(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url_or_id)
        return result[0] if result else None
    return url_or_id.strip()

def fetch_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([item['text'] for item in transcript])
    except Exception as e:
        return f"Error: {e}"

def clean_text(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    summarizer = load_summarizer()
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    return summary

if video_input:
    st.write("Input received!")
    video_id = get_video_id(video_input)
    st.write(f"Extracted video id: {video_id}")

    if not video_id:
        st.error("Invalid YouTube URL or ID")
    else:
        with st.spinner("Fetching transcript..."):
            transcript = fetch_transcript(video_id)

        if transcript.startswith("Error"):
            st.error(transcript)
        else:
            clean_transcript = clean_text(transcript)
            st.subheader("Transcript")
            st.write(clean_transcript)

            with st.spinner("Generating summary..."):
                summary = summarize_text(clean_transcript)

            st.subheader("Summary")
            st.write(summary)
