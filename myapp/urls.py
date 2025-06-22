from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('logout/', views.logout_view, name='logout'),
    path('upload/', views.upload_audio, name='upload_audio'),
    path('download/<int:file_id>/', views.download_audio, name='download_audio'),
    path('audio/<int:file_id>/', views.audio_details, name='audio_details'),  # This line
]