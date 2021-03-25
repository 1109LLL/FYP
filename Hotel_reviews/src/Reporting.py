from DataProvider import Getter




getter = Getter()
hotel_list = getter.getHotelList()
dataset = getter.getFullDataset()

name_rating = dataset[['Hotel_Name', 'Average_Score']].copy()
found = name_rating.groupby('Hotel_Name')['Average_Score'].mean()
found = found.reset_index()




for hotel in hotel_list:

    average_rating = found.loc[found['Hotel_Name']==hotel]

    pattern = {
            "hotel_name":hotel,
            "average_rating":average_rating,
            "best_features":None,
            "location":None,
            "favoured_by_nationality":None,
            "areas to improve":None
    }