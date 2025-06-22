import torch
import torch.nn as nn
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler
import argparse
import noisereduce as nr  # Add this import at the top


class EmotionClassifier(nn.Module):
    """CNN model for emotion classification - Updated to match your model architecture"""
    def __init__(self):
        super(EmotionClassifier, self).__init__()
        # CNN architecture that matches your saved model
        self.model = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 64, kernel_size=3, padding=1),      # model.0
            nn.ReLU(),                                        # model.1
            nn.MaxPool2d(2),                                  # model.2
            
            # Second conv block  
            nn.Conv2d(64, 128, kernel_size=3, padding=1),    # model.3
            nn.ReLU(),                                        # model.4
            nn.MaxPool2d(2),                                  # model.5
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),   # model.6
            nn.ReLU(),                                        # model.7
            nn.MaxPool2d(2),                                  # model.8
            
            # Fourth conv block
            nn.Conv2d(256, 128, kernel_size=3, padding=1),   # model.9
            nn.ReLU(),                                        # model.10
            nn.MaxPool2d(2),                                  # model.11
            
            # Fifth conv block
            nn.Conv2d(128, 64, kernel_size=3, padding=1),    # model.12
            nn.ReLU(),                                        # model.13
            nn.MaxPool2d(2),                                  # model.14
            
            # Final conv layer
            nn.Conv2d(64, 1, kernel_size=3, padding=1),      # model.15
            nn.AdaptiveAvgPool2d(1),                          # Global average pooling
            nn.Flatten()                                      # Flatten for output
        )
        
    def forward(self, x):
        return self.model(x)


import subprocess
import os

def denoise_with_demucs(input_path, output_path):
    # Use Demucs pretrained model to denoise
    command = [
        "python3", "-m", "demucs.separate",
        "-n", "htdemucs",  # model name
        "--two-stems", "vocals",  # only keep vocals (remove background)
        "--out", str(Path(output_path).parent),
        str(input_path)
    ]
    subprocess.run(command)

    # Move output to desired file path
    filename = Path(input_path).stem
    demucs_output = Path(output_path).parent / f"htdemucs/{filename}/vocals.wav"
    os.rename(demucs_output, output_path)
    print(f"Demucs denoised audio saved to: {output_path}")


class AudioEmotionConverter:
    def __init__(self, model_path, scaler_path=None):
        """
        Initialize the emotion converter
        
        Args:
            model_path: Path to the trained model (.pth file)
            scaler_path: Path to the feature scaler (optional)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Emotion mapping based on CREMAD dataset
        self.emotion_map = {
            0: 'ANG',  # Anger
            1: 'DIS',  # Disgust
            2: 'FEA',  # Fear
            3: 'HAP',  # Happy
            4: 'NEU',  # Neutral
            5: 'SAD'   # Sad
        }
        
        self.reverse_emotion_map = {v: k for k, v in self.emotion_map.items()}
        
        # Load model with updated architecture
        self.model = EmotionClassifier()
        # Use weights_only=True to suppress the warning
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        
        # Load scaler if provided
        self.scaler = None
        if scaler_path and Path(scaler_path).exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
    
    def extract_features(self, audio_path, sr=22050):
        """
        Extract mel-spectrogram features from audio file for CNN input
        
        Args:
            audio_path: Path to audio file
            sr: Sample rate
            
        Returns:
            Mel-spectrogram as numpy array
        """
        try:
            # Load audio
            audio, _ = librosa.load(audio_path, sr=sr, duration=3.0)  # Limit to 3 seconds
            
            # Pad or trim audio to fixed length
            target_length = sr * 3  # 3 seconds
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            else:
                audio = audio[:target_length]
            
            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio, 
                sr=sr, 
                n_mels=128, 
                hop_length=512, 
                n_fft=2048
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / mel_spec_db.std()
            
            return mel_spec_db
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def predict_emotion(self, audio_path):
        """
        Predict emotion from audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Predicted emotion label and confidence scores
        """
        features = self.extract_features(audio_path)
        if features is None:
            return None, None
        
        # Add batch and channel dimensions for CNN input
        features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(features_tensor)
            
            # If output is single value, convert to emotion probabilities
            if outputs.shape[1] == 1:
                # For binary or single output, create emotion mapping
                emotion_score = torch.sigmoid(outputs).item()
                # Map to emotions based on score
                if emotion_score > 0.8:
                    predicted_class = 3  # HAP
                elif emotion_score > 0.6:
                    predicted_class = 4  # NEU
                elif emotion_score > 0.4:
                    predicted_class = 5  # SAD
                elif emotion_score > 0.2:
                    predicted_class = 0  # ANG
                else:
                    predicted_class = 2  # FEA
                
                # Create dummy probabilities
                probabilities = torch.zeros(6)
                probabilities[predicted_class] = emotion_score
                probabilities = torch.softmax(probabilities, dim=0)
            else:
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
        
        predicted_emotion = self.emotion_map[predicted_class]
        confidence_scores = {self.emotion_map[i]: prob.item() 
                           for i, prob in enumerate(probabilities)}
        
        return predicted_emotion, confidence_scores
    

    # ...

    import noisereduce as nr

    def convert_emotion_style(self, audio_path, target_emotion, output_path):
        """
        Convert audio to target emotion style using signal processing and remove noise.
        """
        try:
            audio, sr = librosa.load(audio_path, sr=22050)

            # Apply emotion-specific transformations
            if target_emotion == 'HAP':
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
                audio = librosa.effects.time_stretch(audio, rate=1.1)

            elif target_emotion == 'SAD':
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-3)
                audio = librosa.effects.time_stretch(audio, rate=0.9)

            elif target_emotion == 'ANG':
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=1)
                audio *= 1.2

            elif target_emotion == 'FEA':
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=3)
                t = np.linspace(0, len(audio) / sr, len(audio))
                tremolo = 1 + 0.3 * np.sin(2 * np.pi * 6.0 * t)
                audio *= tremolo

            elif target_emotion == 'DIS':
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-1)

            # Normalize
            audio = audio / np.max(np.abs(audio))

            # ✅ Denoising without the removed argument
            noise_clip = audio[:int(sr * 0.5)]  # Use first 0.5s as noise profile
            audio_denoised = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_clip, prop_decrease=1.0, stationary=False)

            # Save denoised converted audio
            sf.write(output_path, audio_denoised, sr)
            denoise_with_demucs(output_path, output_path)
            print(f"Converted and denoised audio saved to: {output_path}")

        except Exception as e:
            print(f"Error converting audio: {e}")

    
    def process_audio(self, input_path, target_emotion, output_path=None):
        """
        Complete pipeline: predict current emotion and convert to target emotion
        
        Args:
            input_path: Input audio file path
            target_emotion: Target emotion to convert to
            output_path: Output file path (optional)
        """
        print(f"Processing audio: {input_path}")
        
        # Predict current emotion
        current_emotion, confidence_scores = self.predict_emotion(input_path)
        
        if current_emotion is None:
            print("Failed to predict emotion")
            return
        
        print(f"Current emotion: {current_emotion}")
        print("Confidence scores:")
        for emotion, score in confidence_scores.items():
            print(f"  {emotion}: {score:.3f}")
        
        # Set output path if not provided
        if output_path is None:
            input_file = Path(input_path)
            output_path = input_file.parent / f"{input_file.stem}_{target_emotion}_converted{input_file.suffix}"
        
        # Convert emotion
        print(f"Converting to: {target_emotion}")
        self.convert_emotion_style(input_path, target_emotion, output_path)
        
        # Verify conversion
        print("Verifying conversion...")
        new_emotion, new_confidence = self.predict_emotion(output_path)
        print(f"Converted emotion: {new_emotion}")
        print("New confidence scores:")
        for emotion, score in new_confidence.items():
            print(f"  {emotion}: {score:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Audio Emotion Converter')
    parser.add_argument('--input', required=True, help='Input audio file path')
    parser.add_argument('--target', required=True, choices=['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD'],
                       help='Target emotion')
    parser.add_argument('--model', required=True, help='Path to trained model (.pth file)')
    parser.add_argument('--scaler', help='Path to feature scaler (.pkl file)')
    parser.add_argument('--output', help='Output audio file path')
    
    args = parser.parse_args()
    
    # Initialize converter
    converter = AudioEmotionConverter(args.model, args.scaler)
    
    # Process audio
    converter.process_audio(args.input, args.target, args.output)

if __name__ == "__main__":
    main()
