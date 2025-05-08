import os
from dotenv import load_dotenv
from google.genai import types



import streamlit as st
import tempfile
import torch
from transformers import (
    AutoProcessor,
    Wav2Vec2ForCTC,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
import requests
from pydub import AudioSegment
import ffmpeg
import librosa
import re

from google import genai

# LangChain prompt templates
from langchain_core.prompts import PromptTemplate

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Disable Streamlit file watcher completely
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Optionally ignore specific heavy libraries (can skip if disabling all)
os.environ["STREAMLIT_WATCHER_IGNORE"] = "torch,torchvision,torchaudio"
def is_gemini_enabled():
    key = GEMINI_API_KEY.strip().lower() if GEMINI_API_KEY else ""
    # Consider these as "disabled"/not a real key
    if not key or key in {"false", "no", "0"}:
        return False
    return True

LLM_MODEL_NAME = "microsoft/Phi-4-mini-reasoning"
LLM_MODEL_ALIAS = LLM_MODEL_NAME.split("/")[-1] if "/" in LLM_MODEL_NAME else LLM_MODEL_NAME

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

st.set_page_config(
    page_title="Minus - Video Summarizer",
    page_icon="📝",
    layout="wide"
)

st.title("Minus - Video Summarizer (Gemini API or Local Inference)")
st.markdown(f"""
This app extracts audio from videos, transcribes the speech, and generates a **clear, AI-polished summary**.
Upload a video file, provide a YouTube link, or enter a direct video URL.

**Summarization is performed using Google's Gemini API if available, otherwise local Phi-4-mini reasoning model.**
""")

@st.cache_resource
def load_asr_model():
    model_name = "facebook/wav2vec2-base-960h"
    st.info("Loading speech-to-text model (wav2vec2)...")
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    st.success("Speech-to-text model loaded.")
    return {"model": model, "processor": processor}

@st.cache_resource
def load_local_llm():
    st.info(f"Loading local LLM: {LLM_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    st.success("Local LLM loaded.")
    return generator

def extract_audio_from_youtube(youtube_url):
    try:
        if yt_dlp is None:
            raise ImportError("yt-dlp is not installed. Please install it with 'pip install yt-dlp'")
        st.info("Step 1: Downloading YouTube video audio. Please wait...")
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "audio.%(ext)s")
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': audio_path,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            audio_file = ydl.prepare_filename(info)
            audio_file = os.path.splitext(audio_file)[0] + ".mp3"
        st.success("YouTube audio downloaded.")
        st.info("Step 2: Converting audio to WAV format...")
        audio = AudioSegment.from_file(audio_file)
        wav_path = os.path.join(temp_dir, "audio.wav")
        audio.export(wav_path, format="wav")
        st.success("Audio conversion to WAV completed.")
        return wav_path, info.get("title", "YouTube Video")
    except Exception as e:
        st.error(f"Error extracting audio from YouTube: {str(e)}")
        return None, None

def extract_audio_from_web(video_url):
    try:
        st.info("Step 1: Downloading video from the provided web link. Please wait...")
        response = requests.get(video_url, stream=True)
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, "video.mp4")
        with open(video_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
        st.success("Web video downloaded.")
        st.info("Step 2: Extracting and converting audio to WAV format...")
        wav_path = os.path.join(temp_dir, "audio.wav")
        (
            ffmpeg
            .input(video_path)
            .output(wav_path, acodec='pcm_s16le', ac=1, ar='16k')
            .run(quiet=True, overwrite_output=True)
        )
        st.success("Audio extraction and conversion completed.")
        return wav_path, "Web Video"
    except Exception as e:
        st.error(f"Error extracting audio from web video: {str(e)}")
        return None, None

def extract_audio_from_upload(uploaded_file):
    try:
        st.info("Step 1: Processing and saving the uploaded video file...")
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, "uploaded_video")
        with open(video_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.success("Video file saved.")
        st.info("Step 2: Extracting and converting audio to WAV format...")
        wav_path = os.path.join(temp_dir, "audio.wav")
        (
            ffmpeg
            .input(video_path)
            .output(wav_path, acodec='pcm_s16le', ac=1, ar='16k')
            .run(quiet=True, overwrite_output=True)
        )
        st.success("Audio extraction and conversion completed.")
        return wav_path, uploaded_file.name
    except Exception as e:
        st.error(f"Error extracting audio from uploaded file: {str(e)}")
        return None, None

def transcribe_audio(audio_path):
    try:
        st.info("Step 3: Transcribing audio to text. This may take a few moments...")
        asr = load_asr_model()
        model = asr["model"]
        processor = asr["processor"]
        audio_array, _ = librosa.load(audio_path, sr=16000)
        max_audio_len = 16000 * 60 * 5  # 5 minutes max at 16kHz
        audio_array = audio_array[:max_audio_len]
        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        st.success("Transcription completed.")
        return transcription
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return None

def summarize_with_gemini(text, max_tokens=1024):
    try:
        st.info("Generating summary with Gemini API...")
        # Use PromptTemplate for the Gemini prompt
        prompt_template = PromptTemplate.from_template(
            "Summarize the following transcript in clear English, focusing on all main ideas. "
            "Do NOT use bullet points, labels, tags, or introductory phrases. Only output the summary, nothing else.\n\n"
            "Transcript:\n{transcript}\n"
        )
        prompt_value = prompt_template.invoke({"transcript": text})
        prompt = str(prompt_value)
        model = genai.Client(api_key=GEMINI_API_KEY)
        response = model.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=0.3,
            ),
        )
        summary = getattr(response, "text", None) or str(response)
        summary = re.sub(r"<.*?>", "", summary)
        st.success("Gemini summary completed.")
        return summary
    except Exception as e:
        st.warning(f"Gemini API unavailable or failed: {str(e)}")
        return None

def summarize_with_local(text, max_length=1024):
    try:
        st.info("Generating summary with local LLM...")
        generator = load_local_llm()
        # Use a LangChain PromptTemplate for prompt construction
        prompt_template = PromptTemplate.from_template(
            "Summarize the following transcript in clear English, focusing on all main ideas. "
            "Do NOT use bullet points, labels, tags, or introductory phrases. Only output the summary, nothing else.\n\n"
            "Transcript:\n{transcript}\n"
        )
        prompt_value = prompt_template.invoke({"transcript": text})
        prompt = str(prompt_value)  # get formatted string prompt

        result = generator(
            prompt,
            max_length=max_length,
            do_sample=False,
            truncation=True
        )
        summary = result[0]['generated_text']
        st.success("Local LLM summary completed.")
        return summary
    except Exception as e:
        st.error(f"Error summarizing with local model: {str(e)}")
        return None

def main():
    input_option = st.radio(
        "Select input type:",
        ["YouTube Link", "Web Video Link", "Upload Video File"]
    )

    # Add a slider for summary length
    st.markdown("#### Maximum Summary Length (tokens/words)")
    max_summary_length = st.slider(
        "Choose the maximum length of the summary",
        min_value=128, max_value=2048, value=1024, step=64,
        help="This controls the maximum length of the generated summary"
    )

    audio_path = None
    video_title = None

    if input_option == "YouTube Link":
        youtube_url = st.text_input("Enter YouTube URL:")
        if youtube_url and st.button("Process YouTube Video"):
            audio_path, video_title = extract_audio_from_youtube(youtube_url)

    elif input_option == "Web Video Link":
        web_url = st.text_input("Enter Web Video URL:")
        if web_url and st.button("Process Web Video"):
            audio_path, video_title = extract_audio_from_web(web_url)

    elif input_option == "Upload Video File":
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file and st.button("Process Uploaded Video"):
            audio_path, video_title = extract_audio_from_upload(uploaded_file)

    if audio_path:
        with st.spinner("Processing all steps..."):
            if video_title:
                st.subheader(f"Processing: {video_title}")
            transcription = transcribe_audio(audio_path)
            if transcription:
                with st.expander("Show Full Transcription"):
                    st.write(transcription)
                summary = None
                if is_gemini_enabled():
                    summary = summarize_with_gemini(transcription, max_tokens=max_summary_length)
                    if not summary:
                        st.warning("Falling back to local LLM for summarization...")
                        summary = summarize_with_local(transcription, max_length=max_summary_length)
                else:
                    summary = summarize_with_local(transcription, max_length=max_summary_length)

                if summary:
                    st.subheader("AI-Polished Summary")
                    st.markdown(f"> {summary}")
                    st.success("All steps completed! Download your transcription and summary below.")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "Download Transcription",
                            transcription,
                            file_name="transcription.txt",
                            mime="text/plain"
                        )
                    with col2:
                        st.download_button(
                            "Download Summary",
                            summary,
                            file_name="summary.txt",
                            mime="text/plain"
                        )

if __name__ == "__main__":
    main()