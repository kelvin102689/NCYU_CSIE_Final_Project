from django import forms
from .models import Video
import os

#50MB
MAX_UPLOAD_SIZE = "52428800"

class UploadVideoForm(forms.ModelForm):
    class Meta :
        model = Video
        fields = ('video',)
        widgets = {
            #'class': 'form-control', 'style': 'width:30%;'
            'video': forms.FileInput(
                attrs={'class': 'btn btn-outline-dark border-secondary','style': 'width:70%;'})
        }
        labels = {'video' : ''}
        help_texts = {
            'video' : ''
        }

    def check_type_and_size(self):
        # extract the username and text field from the data
        video = self.cleaned_data.get('video')
        if video == None :
            raise forms.ValidationError('Missing video file')
        try:
            extension = os.path.splitext(video.name)[1]
            if extension != '.mp4' or video.size > int(MAX_UPLOAD_SIZE):
                print(extension)
                print(video.size)
                return "Invalid"
        except Exception as e:
            raise forms.ValidationError('Can not identify file type')
        return "Valid"

