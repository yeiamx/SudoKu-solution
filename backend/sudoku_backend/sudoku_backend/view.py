from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def index(request):
    context = {}
    #context['hello'] = 'Hello World! (draw version)'
    #context['imgUrl'] = ''

    return render(request, 'index.html', context)

@csrf_exempt
def solve(request):
    if request.method == 'POST':
        #print(request.POST.get('data'))

        dataStr = str(request.POST.get('data'))

        return HttpResponse(json.dumps({
                "status": 'success',
                "result": 'data:image/png;base64,'+dataStr
            }))
