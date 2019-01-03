# Notes for Networking

**To copy files to google cloud instance:**

```
gcloud compute scp filename instance-1:/TravelApp --recurse (other tags)
```
Sometimes we may need to change the default project before uploading material.  This is accomplished with:

```
gcloud config set core/project projectname
```

Note that the 'core' here is optional.  In general, the pattern for setting gcloud properties is
```
gcloud config set SECTION/PROPERTY VALUE --flags
```


** Upload to google bucket **

```
gsutil cp [LOCAL_OBJECT_LOCATION] gs://[DESTINATION_BUCKET_NAME]/
```

** To download stuff from google bucket in python **

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
s=blob1.download_as_string()
usable_stuff=pkl.loads(s)
```
