# Minus

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![GitHub Repo](https://img.shields.io/badge/GitHub-muhammad--fiaz%2FMinus-blue?logo=github)](https://github.com/muhammad-fiaz/Minus)

Minus is a Streamlit app that extracts audio from videos, transcribes speech, and summarizes content using the Gemini API or a local LLM for clear, concise summaries.

## Features

- **Audio Extraction:** Upload video files and extract high-quality audio.
- **Speech Transcription:** Converts extracted audio to text using advanced speech-to-text models.
- **Summarization:** Summarizes transcribed content using:
  - Google Gemini API (cloud-based)
  - Local LLM (runs on your machine)
- **User-Friendly Interface:** Powered by [Streamlit](https://streamlit.io/) for easy and interactive use.

## Getting Started

### Prerequisites

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- Gemini API key (if using Gemini summarization)
- [ffmpeg](https://ffmpeg.org/) installed (for audio extraction)
- Dependencies as listed in `requirements.txt`

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/muhammad-fiaz/Minus.git
   cd Minus
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys and configuration:**
   - Add your Gemini API key and configure paths as needed (see below).

### Usage

Run the Streamlit app:

```bash
streamlit run main.py
```

1. Upload a video file.
2. Choose the transcription and summarization method (Gemini API or local LLM).
3. View and copy your summary!

## Configuration

- **Gemini API:** Set your API key as an environment variable or in a config file.
- **Local LLM:** Ensure the required model files are available locally (see documentation).

## Project Structure

- `Main.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `README.md` - This file

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Contributing

Contributions are welcome! Please open an issue or pull request on [GitHub](https://github.com/muhammad-fiaz/Minus).

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Gemini API](https://ai.google.dev/gemini-api/docs)
- Open-source Hugging Face models for local LLM