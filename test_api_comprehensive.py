#!/usr/bin/env python3
"""
Comprehensive test for the Zoom Meeting Transcript API
Tests the direct download functionality
"""

import requests
import os
import time
from pathlib import Path

API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data}")
            return data.get('model_loaded', False)
        else:
            print(f"❌ Health check failed: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        print("   Make sure the server is running on http://localhost:8000")
        return False

def test_transcription_with_file(audio_file_path):
    """Test transcription with a specific file"""
    if not os.path.exists(audio_file_path):
        print(f"❌ Audio file not found: {audio_file_path}")
        return False
    
    file_size = os.path.getsize(audio_file_path) / (1024 * 1024)  # Size in MB
    print(f"🎵 Testing transcription with: {audio_file_path} ({file_size:.2f} MB)")
    
    try:
        with open(audio_file_path, 'rb') as f:
            files = {'file': (os.path.basename(audio_file_path), f)}
            data = {'language': 'en'}
            
            print("📤 Uploading file and processing...")
            start_time = time.time()
            
            response = requests.post(
                f"{API_BASE_URL}/transcribe", 
                files=files, 
                data=data,
                timeout=300  # 5 minutes timeout for processing
            )
            
            processing_time = time.time() - start_time
            print(f"⏱️  Processing time: {processing_time:.1f} seconds")
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Transcription successful!")
            
            # Check if we got a file
            content_type = response.headers.get('content-type', '')
            if 'application/vnd.openxmlformats-officedocument' in content_type:
                # Extract filename from content-disposition header
                content_disposition = response.headers.get('content-disposition', '')
                if 'filename=' in content_disposition:
                    filename = content_disposition.split('filename=')[1].strip('"')
                else:
                    # Fallback filename
                    original_name = Path(audio_file_path).stem
                    filename = f"transcript_{original_name}.docx"
                
                # Save the transcript file
                with open(filename, 'wb') as f:
                    f.write(response.content)
                
                file_size_kb = len(response.content) / 1024
                print(f"✅ Transcript saved as: {filename}")
                print(f"📄 File size: {file_size_kb:.1f} KB")
                print(f"📊 Processing rate: {file_size / processing_time:.2f} MB/min")
                return True
            else:
                print(f"❌ Unexpected response type: {content_type}")
                print(f"Response: {response.text[:200]}...")
                return False
        else:
            print(f"❌ Transcription failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (processing took too long)")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def find_audio_files():
    """Find available audio files for testing"""
    extensions = ['.m4a', '.mp4', '.wav', '.mp3', '.flac', '.ogg']
    audio_files = []
    
    for ext in extensions:
        files = list(Path('.').glob(f'*{ext}'))
        audio_files.extend(files)
    
    return sorted(audio_files)

def main():
    """Main test function"""
    print("🚀 Comprehensive Zoom Meeting Transcript API Test")
    print("=" * 60)
    
    # Test health check
    if not test_health_check():
        print("\n❌ Server is not running or not ready!")
        print("   Start the server with: python3 main.py")
        return
    
    print("\n" + "=" * 60)
    
    # Find audio files
    audio_files = find_audio_files()
    
    if audio_files:
        print(f"📁 Found {len(audio_files)} audio file(s):")
        for i, file_path in enumerate(audio_files, 1):
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            print(f"   {i}. {file_path} ({file_size:.2f} MB)")
        
        print("\n" + "=" * 60)
        
        # Test with the first file
        success = test_transcription_with_file(str(audio_files[0]))
        
        if success:
            print("\n🎉 API test completed successfully!")
        else:
            print("\n❌ API test failed!")
    else:
        print("📝 No audio files found for testing.")
        print("\nTo test the API:")
        print("1. Place an audio file (m4a, mp4, wav, mp3, etc.) in this directory")
        print("2. Run this script again")
        print("3. Or use Postman with the provided collection")
        print("\nYou can also create test audio files with:")
        print("   python3 create_test_audio.py")

if __name__ == "__main__":
    main()
