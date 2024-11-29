"""
- Convert the parsed logs into number sequences for DeepLog.     
"""

import pandas as pd
from collections import defaultdict


def generate_sequence_data(log_file, template_file):
    templates = pd.read_csv(template_file)
    event_id_map = {}
    for i, row in templates.iterrows():
        event_id = row["EventId"]
        event_id_map[event_id] = i
    
    # Group the logs based on the date
    logs = pd.read_csv(log_file)
    log_map = defaultdict(list)
    for _, row in logs.iterrows():
        event_id, date, time = row["EventId"], row["Date"], row["Time"]
        event_id_encoded = str(event_id_map[event_id])
        key = generate_key(date, time)
        log_map[key].append(event_id_encoded)
    
    with open("./data/hdfs_logs", "w") as text_file:
        for key in log_map:
            sequence = " ".join(log_map[key])
            sequence += "\n"
            text_file.write(sequence)

def generate_key(date, time):
    time_arr = time.split(":")
    return (date, int(time_arr[0]), int(time_arr[1]), int(float(time_arr[2])))

if __name__=="__main__":
    template_file = "../anomalies_microservice_trainticket_version_configurations/ts-auth-mongo_4.4.15_2022-07-13/LOGS_ts-auth-service_3_Mongo:4.4.15.txt_templates.csv"
    log_file = "../anomalies_microservice_trainticket_version_configurations/ts-auth-mongo_4.4.15_2022-07-13/LOGS_ts-auth-service_3_Mongo:4.4.15.txt_structured.csv"
    generate_sequence_data(log_file, template_file)
    