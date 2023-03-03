from django.urls import path
from . import views  #引用這個資料夾中的views檔案


urlpatterns = [
    path('', views.UploadVideo, name = "UploadVideo"),
    path('test', views.Detection, name = "Detection"),
]

#urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)