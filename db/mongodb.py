import pymongo
import yaml
from db.conf.conf import conf

try:
    mongo_client = pymongo.MongoClient(conf["mongodb"]["uri"])
    mongo_db = mongo_client["deepfake"]
    segment_db = mongo_db["segment"]
    analysis_db = mongo_db["analysis"]

    logs.info(f"mongoDB has already been connected: {mongo_db.name}")
except pymongo.errors.ConnectionFailure as e:
    logs.error(f"Failed to connect to mongoDB: {e}")
    raise e

def insert_segment(segment: Dict[str, Any]) -> None:
    segment_db.insert_one(segment)

def get_segment(segment_id: str) -> Dict[str, Any]:
    return segment_db.find_one({"_id": segment_id})

def insert_analysis(analysis: Dict[str, Any]) -> None:
    analysis_db.insert_one(analysis)

def get_analysis(analysis_id: str) -> Dict[str, Any]:
    return analysis_db.find_one({"_id": analysis_id})

def get_segment_by_task_id(task_id: str) -> Dict[str, Any]:
    return segment_db.find_one({"task_id": task_id})

def get_analysis_by_task_id(task_id: str) -> Dict[str, Any]:
    return analysis_db.find_one({"task_id": task_id})