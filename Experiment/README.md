
# ANRL-UCI-Test-Networks

  

## How to run on Google Cloud:

Download Google Cloud SDK and CLI

Setup Google Cloud credentials using gcloud init, must have API access to Google AI Platform enabled on GCP account.

Run `sh gcloud_train.sh` to start training on gcp servers, can specify main python driver file to run based on experiment variable in shell script.

Run `sh local_train.sh` to run code locally, can also specify main python driver to run based on variables on experiment variable in shell script

Experiment variable in both shell scripts looks like this: `experiment="pythonfile"`
**Note**: the specified python driver file should not include the `.py` extension.
 ### Google Cloud AI Platform Setup Documentation
 [https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-keras](https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-keras)

# Potential issues

  

## Running locally

Current issues are that you cannot locally run via python command because current python package import setup is configured for Google Cloud.

In order to fix issue, you have to remove Experiment from all the file imports.

ex: 
`from Experiment.cnn_deepFogGuard import define_deepFogGuard_CNN` to `from cnn_deepFogGuard import define_deepFogGuard_CNN`.

  

## Saving output files and text files

In this program, there are mentions of `gs://anrl-storage/*` like `gs://anrl-storage/results/` or `gs://anrl-storage/models`.

`gs://anrl-storage` is a Google Cloud storage bucket that I made to store results. If you want to store in my bucket, you will need to give me your service account number tied to your Google Cloud AI Platform so I can add you as a user of the bucket.

If you do not want to deal with the issues of bucket verification, you can make your own bucket. After making your own bucket, make folders named results, data, and models.

ex:

	your_own_bucket_name

		-/results

		-/models

		-/data

  

`/results` keeps all of the result text files from training.

`/models` keeps all of the saved models after training.

`/data` keeps all of the experiment datasets, mHealth is the only dataset in this folder for now.

  

## Loading dataset from Google Cloud Storage

In order to load dataset from Google Cloud Storage, you will have to use this command:

`os.system('gsutil -m cp -r gs://anrl-storage/data/mHealth_complete.log ./')`

You will replace gs://anrl-storage/data/mHealth_complete.log with where your data is located in your bucket.
This command loads the data from your Google Storage bucket to the VM that is running the python code. 