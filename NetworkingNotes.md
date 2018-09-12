# Notes for Networking

**To copy files to google cloud instance:**

gcloud compute scp filename instance-1:/TravelApp --recurse (other tags)

** Upload to google bucket **

gsutil cp [LOCAL_OBJECT_LOCATION] gs://[DESTINATION_BUCKET_NAME]/

** To download stuff from google bucket **

```python
import google.cloud.storage
import os
import pickle as pkl
storage_client = google.cloud.storage.Client("TravelApp")

bucket_name = 'travelapp_luxingshifu'
bucket = storage_client.get_bucket(bucket_name)

file = 'remote_filename'

blob1 = bucket.blob(os.path.basename(file))
blob1.download_to_filename('local_file')
```
