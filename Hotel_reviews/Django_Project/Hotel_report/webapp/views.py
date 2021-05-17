from django.shortcuts import render

from django.http import HttpResponse
from django.views.generic import ListView, DetailView
from .models import *
import pandas as pd
import pickle

# Create your views here.
def index(request):
    unique_hotel_list = getHotels()
    context = {
        'hotel_names':unique_hotel_list
    }
    return render(request, 'index.html', context)

def report(request):
    hotel_selected = request.GET.get('select')
    
    # full_dataset = pd.read_csv("../../data/Hotel_Reviews.csv")

    # average rating
    average_rating = get_average_ratings(hotel_selected)

    # address
    address = get_hotel_address(hotel_selected)

    # favoured and disliked by customers from
    pos_nationality_score, neg_nationality_score = get_nationality_preferences(hotel_selected)

    # feature comments
    feature_info, worst_features = get_feature_comments(hotel_selected)

    # content to render
    context = {
        'hotel_name':hotel_selected,
        'features':feature_info,
        'worst_features':worst_features,
        'average_rating':average_rating,
        'location':address,
        'pos_nationality_score':pos_nationality_score,
        'neg_nationality_score':neg_nationality_score
    }

    return render(request, 'report.html', context)    

