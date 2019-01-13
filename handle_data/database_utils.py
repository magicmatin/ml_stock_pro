from pymongo import MongoClient
DB_CONN = MongoClient("mongodb://192.168.56.101:27017")["quant_01"]
