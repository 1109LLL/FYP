from django.shortcuts import render

from django.http import HttpResponse
from django.views.generic import ListView, DetailView
import pandas as pd

# Create your views here.
def index(request):
    df = pd.read_csv("../../data/Hotel_Reviews.csv", usecols=['Hotel_Name'])
    unique_hotel_list = df.Hotel_Name.unique()
    context = {
        'hotel_names':unique_hotel_list
    }
    return render(request, 'index.html', context)

def report(request):
    hotel_selected = request.GET.get('select')
    
    full_dataset = pd.read_csv("../../data/Hotel_Reviews.csv")

    # average rating
    name_rating = full_dataset[['Hotel_Name', 'Average_Score']].copy()
    name_rating = name_rating.loc[name_rating['Hotel_Name']==hotel_selected]
    average_rating = name_rating['Average_Score'].unique().tolist()

    # address
    name_address = full_dataset[['Hotel_Name', 'Hotel_Address']].copy()
    name_address = name_address.loc[name_address['Hotel_Name']==hotel_selected]
    address = name_address['Hotel_Address'].unique().tolist()

    # favoured by customers from
    pos_name_nationality_score = full_dataset[['Hotel_Name', 'Reviewer_Nationality', 'Reviewer_Score']].copy()
    pos_name_nationality_score = pos_name_nationality_score.loc[pos_name_nationality_score['Hotel_Name']==hotel_selected]
    pos_nationality_score = pos_name_nationality_score.groupby('Reviewer_Nationality', as_index=False).mean().sort_values(by='Reviewer_Score', ascending=False)
    pos_nationality_score = pos_nationality_score.loc[pos_nationality_score['Reviewer_Score'] >= 8,:]
    pos_nationality_score = pos_nationality_score.values.tolist()
    
    # disliked by customers from
    neg_name_nationality_score = full_dataset[['Hotel_Name', 'Reviewer_Nationality', 'Reviewer_Score']].copy()
    neg_name_nationality_score = neg_name_nationality_score.loc[neg_name_nationality_score['Hotel_Name']==hotel_selected]
    neg_nationality_score = neg_name_nationality_score.groupby('Reviewer_Nationality',as_index=False).mean().sort_values(by='Reviwer_Score', ascending=False)
    neg_nationality_score = neg_nationality_score.loc[neg_nationality_score['Reviewer_Score'] <= 2,:]
    neg_nationality_score = neg_nationality_score.values.tolist()

    # feature comments
    features = ["room","bed","location","bathroom","staff","staircase",
                    "park","hotel","building","style","transport","parking",
                    "food","breakfast","lunch","dinner","restaurant", "bar"]

    feature_info = []
    for feature in features:
        df = pd.read_csv("../../distance_match_labelled_hotels/{}.csv".format(feature))
        feature_of_hotel = df.loc[df['Hotel_Name']==hotel_selected]
        if feature_of_hotel.empty:
            continue

        feature_of_hotel = feature_of_hotel["Descriptions"]
        subset = feature_of_hotel.values.tolist()
        
        if subset[0] == '[]':
            continue

        info = []
        info.append(feature)
        info.append(subset)
        feature_info.append(info)

    # content to render
    context = {
        'hotel_name':hotel_selected,
        'features':feature_info,
        'average_rating':average_rating,
        'location':address,
        'pos_nationality_score':pos_nationality_score,
        'neg_nationality_score':neg_nationality_score
    }

    return render(request, 'report.html', context)    