import base64
import sqlite3
import os
import numpy as np
from pkg_resources import resource_stream


def dict_factory(cursor, row):  
    d = {}  
    for idx, col in enumerate(cursor.description):  
        d[col[0]] = row[idx]  
    return d 

class DataBase(object):
    def __init__(self, db_path:str):
        db_path = os.path.realpath(db_path)

        if os.path.exists(db_path):
            self.db_path = db_path
        else:
            db_path_dir = os.path.dirname(db_path)
            os.makedirs(db_path_dir, exist_ok=True)
            self.db_path = db_path
        
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = dict_factory
        self.cursor = self.conn.cursor()
        self.init_database()
    
    def init_database(self):
        """
        初始化数据库， 若表不存在则创建
        """
        sql = """
        CREATE TABLE IF NOT EXISTS vprtable (
            `id` INTEGER PRIMARY KEY AUTOINCREMENT,
            `username` TEXT NOT NULL,
            `vector` TEXT NOT NULL,
            `wavpath` TEXT  NOT NULL
            ); 
        """
        self.cursor.execute(sql)
        self.conn.commit()
    
    def execute_base(self, sql, data_dict):
        self.cursor.execute(sql, data_dict)
        self.conn.commit()
    
    def insert_one(self, username, vector_base64:str, wav_path):
        if not os.path.exists(wav_path):
            return None, "wav not exists"
        else:
            sql = f"""
            insert into 
            vprtable (username, vector, wavpath)
            values (?, ?, ?)
            """
            try:
                self.cursor.execute(sql, (username, vector_base64, wav_path))
                self.conn.commit()
                lastidx = self.cursor.lastrowid
                return lastidx, "data insert success"
            except Exception as e:
                print(e)
                return None, e
            
    def select_all(self):
        sql = """
        SELECT * from vprtable
        """
        result = self.cursor.execute(sql).fetchall()
        return result
    
    def select_by_id(self, vpr_id):
        sql = f"""
        SELECT * from vprtable WHERE `id` = {vpr_id}
        """
        result = self.cursor.execute(sql).fetchall()
        return result
    
    def select_by_username(self, username):
        sql = f"""
        SELECT * from vprtable WHERE `username` = '{username}'
        """
        result = self.cursor.execute(sql).fetchall()
        return result

    def drop_by_username(self, username):
        sql = f"""
        DELETE from vprtable WHERE `username`='{username}'
        """
        self.cursor.execute(sql)
        self.conn.commit()
    
    def drop_all(self):
        sql = f"""
        DELETE from vprtable
        """
        self.cursor.execute(sql)
        self.conn.commit()
    
    def drop_table(self):
        sql = f"""
            DROP TABLE vprtable
        """
        self.cursor.execute(sql)
        self.conn.commit()
    
    def encode_vector(self, vector:np.ndarray):
        return base64.b64encode(vector).decode('utf8')
    
    def decode_vector(self, vector_base64, dtype=np.float32):
        b = base64.b64decode(vector_base64)
        vc = np.frombuffer(b, dtype=dtype)
        return vc
    