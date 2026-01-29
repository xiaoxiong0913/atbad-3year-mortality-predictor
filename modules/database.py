import sqlite3
import pandas as pd
import datetime
import json
import os

class PatientDatabase:
    def __init__(self, db_name="patient_records.db"):
        self.db_name = db_name
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        # 创建一个极其通用的表，inputs 存为 JSON 字符串，不再验证具体字段
        c.execute('''
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                inputs TEXT,
                risk_prob REAL,
                risk_label TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def add_record(self, inputs_dict, prob, label):
        try:
            conn = sqlite3.connect(self.db_name)
            c = conn.cursor()
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 无论输入什么 key，都转为 JSON 字符串存储，彻底避免 Validation Error
            inputs_str = json.dumps(inputs_dict)
            
            c.execute("INSERT INTO records (timestamp, inputs, risk_prob, risk_label) VALUES (?, ?, ?, ?)",
                      (timestamp, inputs_str, prob, label))
            conn.commit()
            conn.close()
            print(f"INFO: Record added successfully.")
        except Exception as e:
            print(f"ERROR: Failed to add record to DB: {e}")

    def get_all_records(self):
        try:
            conn = sqlite3.connect(self.db_name)
            df = pd.read_sql_query("SELECT * FROM records ORDER BY timestamp DESC", conn)
            conn.close()
            
            # 尝试解析 JSON 回去，如果失败则返回原始数据
            if not df.empty and 'inputs' in df.columns:
                try:
                    inputs_df = pd.json_normalize(df['inputs'].apply(json.loads))
                    df = pd.concat([df.drop('inputs', axis=1), inputs_df], axis=1)
                except:
                    pass
            return df
        except Exception as e:
            print(f"DB Error: {e}")
            return pd.DataFrame()
