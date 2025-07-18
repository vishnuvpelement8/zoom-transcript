# Zoom Meeting Transcript API

A production-ready FastAPI application that converts audio files from Zoom meetings into formatted transcripts with automatic speaker identification.

## 🚀 Features

### Core Functionality
- **Audio Transcription**: High-quality speech-to-text using OpenAI Whisper
- **Speaker Identification**: Automatic speaker detection and assignment
- **Multiple Formats**: Support for M4A, MP4, WAV, MP3, FLAC, OGG
- **Document Generation**: Professional Word documents with timestamps
- **Conversation Analysis**: Intelligent conversation flow detection

### Production Features
- **Security**: Rate limiting, CORS, security headers, input validation
- **Monitoring**: Structured logging, health checks, performance metrics
- **Scalability**: Async processing, resource management, Docker support
- **Configuration**: Environment-based settings, validation
- **Error Handling**: Comprehensive error responses with proper HTTP codes
- **Documentation**: OpenAPI/Swagger documentation with examples

## 📋 Requirements

- Python 3.11+
- FFmpeg (for audio processing)
- 2GB+ RAM (for Whisper model)
- 1GB+ disk space

## 🛠 Installation & Deployment

### Production Deployment (Vercel - Recommended)

1. **Fork/Clone the repository:**
```bash
git clone <repository-url>
cd zoom-meeting-transcript
```

2. **Deploy to Vercel:**
   - Connect your GitHub repository to Vercel
   - Vercel will automatically detect the FastAPI app
   - Set environment variables in Vercel dashboard (optional)
   - Deploy with one click

3. **Environment Variables (Optional):**
   Set these in your Vercel dashboard under Settings > Environment Variables:
   ```
   WHISPER_MODEL_NAME=base
   MAX_FILE_SIZE=52428800
   LOG_LEVEL=INFO
   DEBUG=false
   ```

### Local Development

1. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure environment (optional):**
```bash
cp .env.example .env
# Edit .env for development settings
```

4. **Run development server:**
```bash
python main.py
```

The API will be available at `http://localhost:8000`

The API will be available at `http://localhost:8000`

## API Endpoints

### POST /transcribe
Upload an audio file and get the transcript file directly.

**Parameters:**
- `file`: Audio file (multipart/form-data)
- `language`: Language code (optional, default: "en")

**Response:**
Returns the transcript as a downloadable Word document (.docx file) with automatic filename based on the original audio file.

## Usage with Postman

### Option 1: Import Collection
1. Import the provided `Zoom_Transcript_API.postman_collection.json` file
2. The collection includes pre-configured requests for all endpoints
3. Update the `base_url` variable if needed (default: http://localhost:8000)

### Option 2: Manual Setup
1. Create a new POST request to `http://localhost:8000/transcribe`
2. In the Body tab, select "form-data"
3. Add a key named "file" and set type to "File"
4. Select your audio file
5. Optionally add a "language" key with value like "en", "es", "fr", etc.
6. Send the request
7. The transcript Word document will be downloaded directly

### Testing the API
Run the test scripts to verify everything works:
```bash
# Basic test
python3 test_api.py

# Comprehensive test with detailed output
python3 test_api_comprehensive.py

# Create test audio files (optional)
python3 create_test_audio.py
```

## Supported Audio Formats

- M4A (Zoom default)
- MP4
- WAV
- MP3
- FLAC
- OGG

## Performance Optimizations

- Audio preprocessing (mono conversion, resampling)
- Silence trimming
- Chunked processing for long files
- Optimized Whisper settings for speed
- Smart speaker identification

## Output

The generated transcript includes:
- Meeting metadata (filename, generation time)
- Timestamped conversation with speaker labels
- Summary statistics (turns per speaker, word counts)
