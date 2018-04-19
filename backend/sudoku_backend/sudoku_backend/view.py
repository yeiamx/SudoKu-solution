from django.http import HttpResponse
from django.shortcuts import render

def index(request):
    context = {}
    #context['hello'] = 'Hello World! (draw version)'
    #context['imgUrl'] = ''

    return render(request, 'index.html', context)