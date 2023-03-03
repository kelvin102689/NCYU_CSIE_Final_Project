from django.shortcuts import render

# Create your views here.
def Introduces(request):
    return render(request, "Introduces/Introduces.html")

def Instruction(request):
    return render(request, "Introduces/Instruction.html")
