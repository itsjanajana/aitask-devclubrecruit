import streamlit as st
import requests
import re
from youtube_transcript_api import YouTubeTranscriptApi

st.title("YouTube Video Summarizer using Hugging Face API")

# === Hugging Face API token placeholder ===
API_TOKEN = "hf_your_token_here"

headers = {
    "Authorization": f"Bearer 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"
}

def get_video_id(url_or_id):
    # Extract 11-char video ID from URL or accept ID as is
    if "youtu" in url_or_id:
        result = re.findall(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url_or_id)
        return result[0] if result else None
    else:
        return url_or_id.strip()

def fetch_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([item['text'] for item in transcript])
        return full_text
    except Exception as e:
        return f"Error fetching transcript: {e}"

def clean_text(text):
    # 1. Remove bracketed text like [music], [applause]
    text = re.sub(r'\[.*?\]', '', text)
    # 2. Remove URLs
    text = re.sub(r'http\S+', '', text)
    # 3. Remove speaker labels (e.g., "John:")
    text = re.sub(r'^\s*[A-Za-z0-9_ ]+:', '', text, flags=re.MULTILINE)
    # 4. Replace newlines with space
    text = re.sub(r'[\r\n]+', ' ', text)
    # 5. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # 6. Fix punctuation spacing
    text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)
    # 7. Remove non-printable/control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    # 8. Remove weird characters
    text = re.sub(r'[^A-Za-z0-9 .,?!\'"-]', '', text)
    # 9. Trim spaces
    return text.strip()

def chunk_text(text, max_chunk_size=800):
    # Splits text into chunks <= max_chunk_size, preferably on sentence end.
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def summarize_with_hf(text):
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and 'summary_text' in data[0]:
            return data[0]['summary_text']
        else:
            return "Error: Unexpected response from API."
    else:
        return f"Error: API request failed with status code {response.status_code}"

def summarize_long_text(text):
    # 1. Clean the text
    cleaned = clean_text(text)
    # 2. Chunk it into manageable pieces for the API
    chunks = chunk_text(cleaned)
    # 3. Summarize each chunk separately
    chunk_summaries = []
    for idx, chunk in enumerate(chunks):
        st.write(f"Summarizing chunk {idx+1} of {len(chunks)}...")
        summary = summarize_with_hf(chunk)
        chunk_summaries.append(summary)
    # 4. Combine chunk summaries into one text
    combined_summary = " ".join(chunk_summaries)
    # Optional: summarize combined summary to condense further if too long
    if len(combined_summary) > 800:
        combined_summary = summarize_with_hf(combined_summary)
    return combined_summary

# Streamlit UI
video_input = st.text_input("Enter YouTube video URL or ID:", placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ")

if video_input:
    video_id = get_video_id(video_input)
    if not video_id:
        st.error("Invalid YouTube URL or ID.")
    else:
        st.write(f"**Video ID:** {video_id}")
        st.write("Fetching transcript...")
        transcript = fetch_transcript(video_id)

        if transcript.startswith("Error"):
            st.error(transcript)
        else:
            st.subheader("Transcript (cleaned)")
            st.write(clean_text(transcript))

            st.write("Generating summary...")
            summary = summarize_long_text(transcript)
            st.subheader("Summary")
            st.write(summary)
