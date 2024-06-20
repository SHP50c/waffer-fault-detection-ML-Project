import os


AWS_S3_BUCKET_NAME = "wafer-fault"
MONGO_DATABASE_NAME = "wafer"
MONGO_COLLECTION_NAME = "waferfault"

TARGET_COLUMN = "Good/Bad"
MONGO_DB_URL="mongodb+srv://SHP50c:alphamikefoxtrot@waferfaultdetection.ftdmtif.mongodb.net/?retryWrites=true&w=majority&appName=WaferFaultDetection"

MODEL_FILE_NAME = "model"
MODEL_FILE_EXTENSION = ".pkl"

artifact_folder =  "artifacts"