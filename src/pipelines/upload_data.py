from pymongo.mongo_client import MongoClient
import pandas as pd
import json

# uniform resource identifier
uri = "mongodb+srv://SHP50c:alphamikefoxtrot@waferfaultdetection.ftdmtif.mongodb.net/?retryWrites=true&w=majority&appName=WaferFaultDetection"

# Create a new client and connect to the server
client = MongoClient(uri)

#create database name and collection name
DATABASE_NAME="wafer"
COLLECTION_NAME="waferfault"

#read the data as a dataframe
df=pd.read_csv(r"D:\my stuff.dll\New Courses\PWSkills\PWSkills Data Science\MLProjectsPW\WaferFaultDetection\notebooks\wafer_23012020_041211.csv")
df=df.drop("Unnamed: 0",axis=1)

#convert the data into json
json_record=list(json.loads(df.T.to_json()).values())

#nowdump the data into database
client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)