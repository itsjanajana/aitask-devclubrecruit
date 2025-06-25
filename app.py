import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import re

# Title of the app
st.title("Simple YouTube Video Summarizer")

# Input: YouTube URL or ID
video_input = st.text_input("Enter YouTube video URL or ID:")

def get_video_id(url_or_id):
    # If user enters a full URL, extract video ID using regex
    if "youtu" in url_or_id:
        import re
        result = re.findall(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url_or_id)
        if result:
            return result[0]
        else:
            return None
    else:
        # Assume it's already the video ID
        return url_or_id.strip()

def fetch_transcript(video_id):
    try:
        # Get transcript list (each item has 'text')
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        # Combine all pieces into one string
        full_text = " ".join([item['text'] for item in transcript])
        return full_text
    except Exception as e:
        return f"Error: {e}"

def clean_text(text):
    # Remove bracketed text like [music]
    text = re.sub(r'\[.*?\]', '', text)
    # Remove extra spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

@st.cache_resource
def load_summarizer():
    # Load HuggingFace summarization model once (cached)
    return pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    summarizer = load_summarizer()
    # Summarize the whole text at once (keep it short for this simple version)
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    return summary

if video_input:
    video_id = get_video_id(video_input)

    if not video_id:
        st.error("Invalid YouTube URL or ID")
    else:
        st.write(f"**Video ID:** {video_id}")

        st.write("Fetching transcript...")
        transcript = fetch_transcript(video_id)

        if transcript.startswith("Error"):
            st.error(transcript)
        else:
            clean_transcript = clean_text(transcript)
            st.subheader("Transcript")
            st.write(clean_transcript)

            st.write("Generating summary...")
            summary = summarize_text(clean_transcript)

            st.subheader("Summary")
            st.write(summary)
