# -*- coding: utf-8 -*-
import sqlite3
import pandas as pd
import datetime
import uuid
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)

class DataValidator:
    """
    验证数据的工具类
    """
    @staticmethod
    def validate_record(data):
        # 简单校验，防止空数据
        if not data:
            return False
        return True

class PatientDatabase:
    """
    数据库管理类 (这就是报错找不到的那个类！)
    """
    def __init__(self, db_name="patient_history.db"):
        self.db_name = db_name
        self.table_name = "clinical_predictions"
        self._init_db()

    def _get_connection(self):
        # Streamlit 多线程环境下必须设置 check_same_thread=False
        return sqlite3.connect(self.db_name, check_same_thread=False)

    def _init_db(self):
        """初始化数据库表结构"""
        conn = self._get_connection()
        cursor = conn.cursor()
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id TEXT PRIMARY KEY,
            timestamp DATETIME,
            gender TEXT,
            bun REAL,
            sbp REAL,
            hgb REAL,
            t_wave INTEGER,
            statins INTEGER,
            risk_prob REAL,
            risk_label TEXT
        );
        """
        try:
            cursor.execute(create_table_sql)
            conn.commit()
        except Exception as e:
            logging.error(f"DB Init Error: {e}")
        finally:
            conn.close()

    def add_record(self, inputs, prob, risk_label):
        """插入一条新记录"""
        record_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 这里的 key 必须和 streamlit_app.py 里的 inputs 字典 key 一致
        data_tuple = (
            record_id,
            timestamp,
            inputs.get('Gender', 'Unknown'),
            inputs.get('BUN(mmol/L)', 0.0),
            inputs.get('SBP(mmHg)', 0.0),
            inputs.get('HGB(g/L)', 0.0),
            inputs.get('T wave  abnormalities', 0),
            inputs.get('Statins', 0),
            float(prob),
            risk_label
        )

        insert_sql = f"""
        INSERT INTO {self.table_name} 
        (id, timestamp, gender, bun, sbp, hgb, t_wave, statins, risk_prob, risk_label)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        conn = self._get_connection()
        try:
            conn.execute(insert_sql, data_tuple)
            conn.commit()
            return True
        except Exception as e:
            logging.error(f"Insert Error: {e}")
            return False
        finally:
            conn.close()

    def fetch_all_records(self):
        """获取所有历史记录"""
        conn = self._get_connection()
        query = f"SELECT * FROM {self.table_name} ORDER BY timestamp DESC"
        try:
            df = pd.read_sql_query(query, conn)
            return df
        except Exception as e:
            logging.error(f"Fetch Error: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def delete_all_records(self):
        """清空数据库"""
        conn = self._get_connection()
        try:
            conn.execute(f"DELETE FROM {self.table_name}")
            conn.commit()
        finally:
            conn.close()
