from django.db import models
from django.core.validators import FileExtensionValidator

# Create your models here.
class Video(models.Model):
    name = models.CharField(max_length=255)  #video name
    create_time = models.DateTimeField(auto_now_add=True, blank=True) #自動存物件被創時的日期
    video = models.FileField(upload_to ='videos')
