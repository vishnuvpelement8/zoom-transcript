#!/usr/bin/env python3
"""
Test script for the Zoom Meeting Transcript API
"""

import requests
import time
import os

API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    response = requests.get(f"{API_BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_transcription(audio_file_path):
    """Test the transcription endpoint"""
    if not os.path.exists(audio_file_path):
        print(f"❌ Audio file not found: {audio_file_path}")
        return False

    print(f"🎵 Testing transcription with: {audio_file_path}")

    with open(audio_file_path, 'rb') as f:
        files = {'file': (os.path.basename(audio_file_path), f, 'audio/m4a')}
        data = {'language': 'en'}

        print("📤 Uploading file and processing...")
        response = requests.post(f"{API_BASE_URL}/transcribe", files=files, data=data)

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        print("✅ Transcription successful!")

        # The response should be the file content directly
        if 'content-disposition' in response.headers:
            # Extract filename from content-disposition header
            content_disposition = response.headers['content-disposition']
            filename = content_disposition.split('filename=')[1].strip('"')

            # Save the transcript file
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"✅ Transcript saved as: {filename}")
            print(f"📄 File size: {len(response.content)} bytes")
            return True
        else:
            # Fallback filename
            output_file = f"transcript_{os.path.basename(audio_file_path)}.docx"
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"✅ Transcript saved as: {output_file}")
            print(f"📄 File size: {len(response.content)} bytes")
            return True
    else:
        print(f"❌ Transcription failed: {response.text}")
        return False

def main():
    """Main test function"""
    print("🚀 Testing Zoom Meeting Transcript API")
    print("=" * 50)
    
    # Test health check
    if not test_health_check():
        print("❌ Health check failed!")
        return
    
    print("\n" + "=" * 50)
    
    # Look for sample audio files
    sample_files = ["sample.m4a", "sample.mp4", "sample.wav", "sample.mp3"]
    
    audio_file = None
    for file_path in sample_files:
        if os.path.exists(file_path):
            audio_file = file_path
            break
    
    if audio_file:
        test_transcription(audio_file)
    else:
        print("📝 No sample audio files found.")
        print("To test transcription, place an audio file named:")
        for file_name in sample_files:
            print(f"  - {file_name}")
        print("\nThen run this script again.")
        print("\nOr use Postman to test the API:")
        print(f"  POST {API_BASE_URL}/transcribe")
        print("  Body: form-data with 'file' field containing your audio file")

if __name__ == "__main__":
    main()
