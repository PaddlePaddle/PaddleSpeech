# vpr Demo 没有使用 mysql 与 muilvs, 仅用于docker演示
import logging
import faiss
from matplotlib import use
import numpy as np
from .sql_helper import DataBase
from .vpr_encode import get_audio_embedding

class VPR:
    def __init__(self, db_path, dim, top_k) -> None:
        # 初始化
        self.db_path = db_path
        self.dim = dim
        self.top_k = top_k
        self.dtype = np.float32
        self.vpr_idx = 0
        
        # db 初始化
        self.db = DataBase(db_path)
        
        # faiss 初始化
        index_ip = faiss.IndexFlatIP(dim)
        self.index_ip = faiss.IndexIDMap(index_ip)
        self.init()
    
    def init(self):
        # demo 初始化，把 mysql中的向量注册到 faiss 中
        sql_dbs = self.db.select_all()
        if sql_dbs:
            for sql_db in sql_dbs:
                idx = sql_db['id']
                vc_bs64 = sql_db['vector']
                vc = self.db.decode_vector(vc_bs64)
                if len(vc.shape) == 1:
                    vc = np.expand_dims(vc, axis=0)
                # 构建数据库
                self.index_ip.add_with_ids(vc, np.array((idx,)).astype('int64'))
            logging.info("faiss 构建完毕")
    
    def faiss_enroll(self, idx, vc):
        self.index_ip.add_with_ids(vc, np.array((idx,)).astype('int64'))
    
    def vpr_enroll(self, username, wav_path):
        # 注册声纹
        emb = get_audio_embedding(wav_path)
        emb = np.expand_dims(emb, axis=0)
        if emb is not None:
            emb_bs64 = self.db.encode_vector(emb)
            last_idx, mess = self.db.insert_one(username, emb_bs64, wav_path)
            if last_idx:
                # faiss 注册
                self.faiss_enroll(last_idx, emb)
        else:
            last_idx, mess = None
        return last_idx
    
    def vpr_recog(self, wav_path):
        # 识别声纹
        emb_search = get_audio_embedding(wav_path)
        
        if emb_search is not None:
            emb_search = np.expand_dims(emb_search, axis=0)
            D, I = self.index_ip.search(emb_search, self.top_k)
            D = D.tolist()[0]
            I = I.tolist()[0]            
            return [(round(D[i] * 100, 2 ), I[i]) for i in range(len(D)) if I[i] != -1]
        else:
            logging.error("识别失败")
            return None
    
    def do_search_vpr(self, wav_path):
        spk_ids, paths, scores = [], [], []
        recog_result = self.vpr_recog(wav_path)
        for score, idx in recog_result:
            username = self.db.select_by_id(idx)[0]['username']
            if username not in spk_ids:
                spk_ids.append(username)
                scores.append(score)
                paths.append("")
        return spk_ids, paths, scores
    
    def vpr_del(self, username):
        # 根据用户username, 删除声纹
        # 查用户ID，删除对应向量
        res = self.db.select_by_username(username)
        for r in res:
            idx = r['id']
            self.index_ip.remove_ids(np.array((idx,)).astype('int64'))
        
        self.db.drop_by_username(username)
    
    def vpr_list(self):
        # 获取数据列表
        return self.db.select_all()
    
    def do_list(self):
        spk_ids, vpr_ids = [], []
        for res in self.db.select_all():
            spk_ids.append(res['username'])
            vpr_ids.append(res['id'])
        return spk_ids, vpr_ids 
    
    def do_get_wav(self, vpr_idx):
         res = self.db.select_by_id(vpr_idx)
         return res[0]['wavpath']
         
    
    def vpr_data(self, idx):
        # 获取对应ID的数据
        res = self.db.select_by_id(idx)
        return res
    
    def vpr_droptable(self):
        # 删除表
        self.db.drop_table()
        # 清空 faiss
        self.index_ip.reset()
        
    

if __name__ == '__main__':
    
    db_path = "../../source/db/vpr.sqlite"
    dim = 192
    top_k = 5
    vpr = VPR(db_path, dim, top_k)
    
    # 准备测试数据
    username = "sss"
    wav_path = r"../../source/demo/demo_16k.wav"
    
    # 注册声纹
    vpr.vpr_enroll(username, wav_path)
    
    # 获取数据
    print(vpr.vpr_list())
    
    # 识别声纹
    recolist = vpr.vpr_recog(wav_path)
    print(recolist)
    
    # 通过 id 获取数据
    idx = recolist[0][1]
    print(vpr.vpr_data(idx))

    # 删除声纹
    vpr.vpr_del(username)
    vpr.vpr_droptable()
    
    
    
    