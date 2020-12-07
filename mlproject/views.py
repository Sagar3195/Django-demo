from django.http import HttpResponse
from django.shortcuts import render
import joblib


def home(request):
    return render(request, "base.html")

def result(request):
    model = joblib.load('mpg_model.pkl')
    lst= []
    lst.append(request.GET['cylinders'])
    lst.append(request.GET['displacement'])
    lst.append(request.GET['horsepower'])
    lst.append(request.GET['weight'])
    lst.append(request.GET['acceleration'])
    lst.append(request.GET['model year'])
    lst.append(request.GET['origin'])

    ans = model.predict([lst])



    return render(request, 'result.html', {'ans':  ans})
