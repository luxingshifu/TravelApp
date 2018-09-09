import boto3
import botocore
import pickle as pkl

BUCKET_NAME="bucketeer-02e46111-d1ef-4f0e-a2f7-a30e0aabf7cc"
PLACE_LIST_KEY="full_place_list.pkl"
PLACE_PROFILE_KEY = "place_profiles.pkl"
MODEL_KEY = "model.pkl"

s3=boto3.resource('s3')


try:
    s3.Bucket(BUCKET_NAME).download_file(PLACE_LIST_KEY, 'model/full_place_list.pkl')
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == "404":
        print("'full_place_list.pkl' does not exist.")
    else:
        raise


try:
    s3.Bucket(BUCKET_NAME).download_file(MODEL_KEY, 'model/model.pkl')
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == "404":
        print("'model.pkl' does not exist.")
    else:
        raise


try:
    s3.Bucket(BUCKET_NAME).download_file(PLACE_PROFILE_KEY, 'model/place_profiles.pkl')
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == "404":
        print("'model.pkl' does not exist.")
    else:
        raise


with open('model/model.pkl','rb') as f:
    model=pkl.load(f)

with open('model/place_profiles.pkl','rb') as f:
    full_profiles=pkl.load(f)

with open('model/full_place_list.pkl','rb') as f:
    full_place_list=pkl.load(f)

# def get_data():
#     return model, full_profiles, full_place_list
