from django.db import models
from django.contrib.auth.models import User
import os

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    firebase_uid = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username}'s Profile"

class AudioFile(models.Model):
    EMOTION_CHOICES = [
        ('ANG', 'Anger'),
        ('DIS', 'Disgust'),
        ('FEA', 'Fear'),
        ('HAP', 'Happy'),
        ('NEU', 'Neutral'),
        ('SAD', 'Sad'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    original_file = models.FileField(upload_to='audio/original/')
    converted_file = models.FileField(upload_to='audio/converted/', blank=True, null=True)
    original_emotion = models.CharField(max_length=3, choices=EMOTION_CHOICES, blank=True)
    target_emotion = models.CharField(max_length=3, choices=EMOTION_CHOICES)
    confidence_scores = models.JSONField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_processed = models.BooleanField(default=False)
    
    def __str__(self):
        return f"{self.user.username} - {self.original_emotion} to {self.target_emotion}"

    def get_original_path(self):
        if self.original_file:
            return self.original_file.path
        return None
    
    def get_converted_path(self):
        if self.converted_file:
            return self.converted_file.path
        return None

    def __str__(self):
        return f"{self.user.username} - {self.original_emotion} to {self.target_emotion}"

    def get_original_path(self):
        """Get the absolute path to the original file"""
        if self.original_file:
            return self.original_file.path
        return None
    
    def get_converted_path(self):
        """Get the absolute path to the converted file"""
        if self.converted_file:
            return self.converted_file.path
        return None
