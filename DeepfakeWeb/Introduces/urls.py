from django.urls import path
from . import views  #引用這個資料夾中的views檔案


urlpatterns = [
    path('', views.Introduces, name = "Introduces"),
    path('Instruction', views.Instruction, name = "Instruction")
]

#urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)