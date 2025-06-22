from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import HttpResponse, FileResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from .models import UserProfile, AudioFile
import pyrebase
import json
import os
import sys
from pathlib import Path

# Add project root to Python path (ONLY ONCE!)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import emotion converter (ONLY ONCE!)
try:
    from emotion_project.emotion_converter import AudioEmotionConverter
    EMOTION_CONVERTER_AVAILABLE = True
    print("✅ AudioEmotionConverter imported successfully!")
except ImportError as e:
    print(f"❌ Error importing AudioEmotionConverter: {e}")
    AudioEmotionConverter = None
    EMOTION_CONVERTER_AVAILABLE = False

# Firebase configuration
const = {
    "apiKey": os.environ.get("FIREBASE_API_KEY"),
    "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN"),
    "databaseURL": os.environ.get("FIREBASE_DATABASE_URL"),
    "projectId": os.environ.get("FIREBASE_PROJECT_ID"),
    "storageBucket": os.environ.get("FIREBASE_STORAGE_BUCKET"),
    "messagingSenderId": os.environ.get("FIREBASE_MESSAGING_SENDER_ID"),
    "appId": os.environ.get("FIREBASE_APP_ID"),
}

firebase = pyrebase.initialize_app(const)
auth = firebase.auth()

# Initialize emotion converter
MODEL_PATH = PROJECT_ROOT / 'emotion_project' / 'models' / 'G_neu2sad_final.pth'
emotion_converter = None

def get_emotion_converter():
    """Initialize emotion converter lazily to avoid startup errors"""
    global emotion_converter
    if emotion_converter is None and EMOTION_CONVERTER_AVAILABLE:
        try:
            if MODEL_PATH.exists():
                emotion_converter = AudioEmotionConverter(str(MODEL_PATH))
                print(f"✅ Emotion converter initialized with model: {MODEL_PATH}")
            else:
                print(f"❌ Model file not found: {MODEL_PATH}")
                return None
        except Exception as e:
            print(f"❌ Error initializing emotion converter: {e}")
            return None
    return emotion_converter

# ADD THE MISSING HOME VIEW
def home(request):
    """Home page - shows different content if user is logged in"""
    return render(request, 'home.html')

def signup_view(request):
    """Handle user registration"""
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        
        try:
            # Create user in Firebase
            firebase_user = auth.create_user_with_email_and_password(email, password)
            
            # Create user in Django
            django_user = User.objects.create_user(
                username=email,
                email=email,
                password=password
            )
            
            # Link them together
            UserProfile.objects.create(
                user=django_user,
                firebase_uid=firebase_user['localId']
            )
            
            messages.success(request, 'Account created successfully!')
            return redirect('login')
            
        except Exception as e:
            messages.error(request, 'Error creating account. Please try again.')
    
    return render(request, 'signup.html')

def login_view(request):
    """Handle user login"""
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        try:
            # Authenticate with Firebase
            firebase_user = auth.sign_in_with_email_and_password(email, password)
            # Log in to Django
            django_user = User.objects.get(email=email)
            login(request, django_user)
            # Store Firebase token in session
            request.session['firebase_token'] = firebase_user['idToken']
            return redirect('dashboard')
        except Exception as e:
            messages.error(request, f'Invalid email or password. ({e})')
    return render(request, 'login.html')

def logout_view(request):
    """Log out user"""
    logout(request)
    if 'firebase_token' in request.session:
        del request.session['firebase_token']
    return redirect('home')

def dashboard_view(request):
    """Enhanced dashboard with emotion conversion functionality"""
    if not request.user.is_authenticated:
        return redirect('login')
    
    # Get user's audio files
    user_files = AudioFile.objects.filter(user=request.user).order_by('-created_at')
    
    context = {
        'user_files': user_files,
        'emotion_choices': AudioFile.EMOTION_CHOICES,  # This line was missing or wrong
        'converter_available': EMOTION_CONVERTER_AVAILABLE
    }
    
    return render(request, 'dashboard.html', context)


def upload_audio(request):
    """Handle audio file upload and emotion conversion"""
    if not request.user.is_authenticated:
        return redirect('login')
    
    if not EMOTION_CONVERTER_AVAILABLE:
        messages.error(request, 'Emotion converter is not available. Please check the setup.')
        return redirect('dashboard')
    
    if request.method == 'POST':
        audio_file = request.FILES.get('audio_file')
        target_emotion = request.POST.get('target_emotion')
        
        if not audio_file or not target_emotion:
            messages.error(request, 'Please provide both audio file and target emotion.')
            return redirect('dashboard')
        
        try:
            # Create AudioFile record
            audio_record = AudioFile.objects.create(
                user=request.user,
                original_file=audio_file,
                target_emotion=target_emotion
            )
            
            # Process the audio file
            process_audio_file(audio_record)
            
            messages.success(request, 'Audio file uploaded and processed successfully!')
            
        except Exception as e:
            messages.error(request, f'Error processing audio: {str(e)}')
    
    return redirect('dashboard')

def process_audio_file(audio_record):
    """Process audio file using your emotion converter"""
    converter = get_emotion_converter()
    if converter is None:
        raise Exception("Emotion converter not available")
    
    try:
        input_path = audio_record.get_original_path()
        
        # Create output filename
        output_filename = f"{audio_record.id}_{audio_record.target_emotion}_converted.wav"
        output_dir = Path(default_storage.location) / 'audio' / 'converted'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename
        
        print(f"Processing audio: {input_path} -> {output_path}")
        
        # Predict current emotion
        current_emotion, confidence_scores = converter.predict_emotion(input_path)
        print(f"Detected emotion: {current_emotion}")
        
        # Convert emotion
        converter.convert_emotion_style(
            input_path, 
            audio_record.target_emotion, 
            str(output_path)
        )
        
        # Check if the output file was actually created
        if output_path.exists():
            print(f"✅ Audio converted successfully: {output_path}")
            
            # Update record with relative path for Django FileField
            relative_path = f'audio/converted/{output_filename}'
            audio_record.original_emotion = current_emotion
            audio_record.confidence_scores = confidence_scores
            audio_record.converted_file = relative_path
            audio_record.is_processed = True
            audio_record.save()
        else:
            # Check if Demucs created a different structure
            demucs_output = output_dir / 'htdemucs' / output_filename.replace('.wav', '') / 'vocals.wav'
            if demucs_output.exists():
                print(f"Found Demucs output, moving file: {demucs_output} -> {output_path}")
                
                # Move the file to the expected location
                import shutil
                shutil.move(str(demucs_output), str(output_path))
                
                # Clean up Demucs directories
                try:
                    demucs_dir = output_dir / 'htdemucs'
                    if demucs_dir.exists():
                        shutil.rmtree(str(demucs_dir))
                except:
                    pass  # Ignore cleanup errors
                
                # Update record
                relative_path = f'audio/converted/{output_filename}'
                audio_record.original_emotion = current_emotion
                audio_record.confidence_scores = confidence_scores
                audio_record.converted_file = relative_path
                audio_record.is_processed = True
                audio_record.save()
            else:
                raise Exception(f"Output file not created: {output_path}")
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        # Mark as failed but save the record
        audio_record.is_processed = False
        audio_record.save()
        raise e


def download_audio(request, file_id):
    """Download converted audio file - FIXED VERSION"""
    if not request.user.is_authenticated:
        return redirect('login')
    
    audio_file = get_object_or_404(AudioFile, id=file_id, user=request.user)
    
    if not audio_file.converted_file:
        messages.error(request, 'Converted file not available.')
        return redirect('dashboard')
    
    try:
        # FIXED: Use the .path property directly from the FieldFile
        file_path = audio_file.converted_file.path
        
        # Check if file exists
        if not os.path.exists(file_path):
            messages.error(request, 'File not found on disk.')
            return redirect('dashboard')
        
        # Return the file for download
        return FileResponse(
            open(file_path, 'rb'),
            as_attachment=True,
            filename=f"{audio_file.get_target_emotion_display()}_converted.wav"
        )
        
    except Exception as e:
        messages.error(request, f'Error downloading file: {str(e)}')
        return redirect('dashboard')



def audio_details(request, file_id):
    """Show detailed information about audio conversion"""
    if not request.user.is_authenticated:
        return redirect('login')
    
    try:
        audio_file = get_object_or_404(AudioFile, id=file_id, user=request.user)
        
        # Debug print (remove after testing)
        print(f"Audio file found: {audio_file}")
        print(f"Original emotion: {audio_file.original_emotion}")
        print(f"Target emotion: {audio_file.target_emotion}")
        print(f"Is processed: {audio_file.is_processed}")
        print(f"Has converted file: {bool(audio_file.converted_file)}")
        
        return render(request, 'audio_details.html', {'audio_file': audio_file})
        
    except Exception as e:
        print(f"Error in audio_details view: {e}")
        messages.error(request, f'Error loading audio details: {str(e)}')
        return redirect('dashboard')

