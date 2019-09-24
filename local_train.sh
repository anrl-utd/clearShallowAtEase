
TRAINER_PACKAGE_PATH="./Experiment"
MAIN_TRAINER_MODULE="Experiment.cifar_failout_rate"
PACKAGE_STAGING_PATH="gs://anrl-storage"

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="Brian_Nguyen_$now"
MODEL_NAME=$JOB_NAME.h5
JOB_DIR="gs://cbis-ddsm-cnn"
REGION="us-central1"
MODE="LOCAL"

gcloud ai-platform local train \
--module-name=$MAIN_TRAINER_MODULE \
--package-path=$TRAINER_PACKAGE_PATH \
--job-dir=$JOB_DIR \
-- \
--mode=$MODE \
--train TRUE \
--model_name=$MODEL_NAME
