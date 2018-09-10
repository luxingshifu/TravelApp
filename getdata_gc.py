import google.cloud.storage
import os
import pickle as pkl
storage_client = google.cloud.storage.Client("TravelApp")

bucket_name = 'travelapp_luxingshifu'
bucket = storage_client.get_bucket(bucket_name)

file1 = 'full_place_list.pkl'
file2 = 'model.pkl'
file3 = 'place_profiles.pkl'

blob1 = bucket.blob(os.path.basename(file1))
blob1.download_to_filename('model/full_place_list.pkl')

blob2 = bucket.blob(os.path.basename(file2))
blob2.download_to_filename('model/model.pkl')

blob3 = bucket.blob(os.path.basename(file3))
blob3.download_to_filename('model/place_profiles.pkl')
