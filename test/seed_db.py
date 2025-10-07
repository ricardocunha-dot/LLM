import csv
from pymongo import MongoClient
import os

MONGO_URI = "mongodb://mongo:27017/"
CANDIDATES_CSV = "candidates.csv"
JOBS_CSV = "jobs.csv"

def seed_collection(client, db_name, collection_name, csv_file):
    db = client[db_name]
    collection = db[collection_name]
    
    try:
        with open(csv_file, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            data = [row for row in reader if any(row.values())] # Ignora linhas completamente vazias
        
        # Converte campos numéricos, com tratamento de erros
        for item in data:
            for key, value in item.items():
                if key in ["min_experience", "experience_years"]:
                    try:
                        item[key] = int(value)
                    except (ValueError, TypeError):
                        print(f"Warning: Could not convert '{value}' to int for key '{key}' in {csv_file}. Setting to 0.")
                        item[key] = 0

        print(f"Deleting existing data in collection '{collection_name}'...")
        collection.delete_many({})
        
        print(f"Inserting {len(data)} documents into '{collection_name}'...")
        if data:
            collection.insert_many(data)
        
        print(f"Collection '{collection_name}' seeded successfully!")
        
    except FileNotFoundError:
        print(f"Error: {csv_file} not found. Skipping seeding.")
    except Exception as e:
        print(f"An error occurred while seeding '{collection_name}': {e}")


if __name__ == "__main__":
    print("Starting DB seeding...")
    client = MongoClient(MONGO_URI)
    
    seed_collection(client, "recruitment_db", "jobs", JOBS_CSV)
    seed_collection(client, "recruitment_db", "candidates", CANDIDATES_CSV)
    
    client.close()
    print("DB seeding process finished.")