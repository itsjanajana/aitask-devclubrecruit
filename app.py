import streamlit as st  # Streamlit for UI
from youtube_transcript_api import YouTubeTranscriptApi  # To get YouTube subtitles
from transformers import pipeline  # HuggingFace pipeline for summarization
import re  # Regex for text cleaning

st.title("Simple YouTube Video Summarizer")  # App title

video_input = st.text_input("Enter YouTube video URL or ID:")  # Input box for URL or ID

def get_video_id(url_or_id):
    # If input is a full URL, extract the video ID using regex
    if "youtu" in url_or_id:
        result = re.findall(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url_or_id)
        return result[0] if result else None  # Return first matched ID or None
    return url_or_id.strip()  # Else, assume input is the video ID and strip spaces

def fetch_transcript(video_id):
    try:
        # Get transcript as list of dicts with 'text'
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        # Join all text parts into one string
        return " ".join([item['text'] for item in transcript])
    except Exception as e:
        return f"Error: {e}"  # Return error string if transcript not found or error occurs

def clean_text(text):
    # Remove bracketed text e.g. [music]
    text = re.sub(r'\[.*?\]', '', text)
    # Replace multiple whitespaces/newlines with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()  # Trim leading/trailing spaces

@st.cache_resource  # Cache the model loading to avoid reloading each time
def load_summarizer():
    # Load summarization pipeline with facebook/bart-large-cnn model
    return pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    summarizer = load_summarizer()  # Get cached summarizer
    # Generate summary with specified max and min length
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    return summary

if video_input:
    st.write("Input received!")  # Confirm input received

    video_id = get_video_id(video_input)  # Extract video ID from input
    st.write(f"Extracted video id: {video_id}")  # Show extracted ID

    if not video_id:
        st.error("Invalid YouTube URL or ID")  # Show error if ID not valid
    else:
        with st.spinner("Fetching transcript..."):  # Show loading spinner
            transcript = fetch_transcript(video_id)  # Fetch transcript text

        if transcript.startswith("Error"):
            st.error(transcript)  # Show error if transcript fetch failed
        else:
            clean_transcript = clean_text(transcript)  # Clean transcript text

            st.subheader("Transcript")  # Subtitle for transcript
            st.write(clean_transcript)  # Show cleaned transcript

            with st.spinner("Generating summary..."):  # Spinner during summarization
                summary = summarize_text(clean_transcript)  # Generate summary

            st.subheader("Summary")  # Subtitle for summary
            st.write(summary)  # Display summary
