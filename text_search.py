import pandas as pd
import numpy as np
from towhee import ops, pipe, DataCollection
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

import time


connections.connect(host='192.168.68.108', port='19530')

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__}运行时间: {end_time - start_time}秒")
        return result
    return wrapper



def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),   
            FieldSchema(name="title_vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="link", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="reading_time", dtype=DataType.INT64),
            FieldSchema(name="publication", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="claps", dtype=DataType.INT64),
            FieldSchema(name="responses", dtype=DataType.INT64)
    ]
    schema = CollectionSchema(fields=fields, description='search text')
    collection = Collection(name=collection_name, schema=schema)
    
    index_params = {
        'metric_type': "L2",
        'index_type': "IVF_FLAT",
        'params': {"nlist": 2048}
    }
    collection.create_index(field_name='title_vector', index_params=index_params)
    return collection



def insert_pipe():
    collection = create_milvus_collection('search_article_in_medium', 768)
    df = pd.read_csv('New_Medium_Data.csv', converters={'title_vector': lambda x: eval(x)})
    insert_pipe = (pipe.input('df')
                   .flat_map('df', 'data', lambda df: df.values.tolist())
                   .map('data', 'res', ops.ann_insert.milvus_client(host='192.168.68.108', 
                                                                    port='19530',
                                                                    collection_name='search_article_in_medium'))
                   .output('res')
    )
    print("start insert")
    insert_pipe(df)
    print("end insert")
    collection.load()
    print(collection.num_entities)


# 预加载模型
dpr_model = ops.text_embedding.dpr(model_name="facebook/dpr-ctx_encoder-single-nq-base")
# 预创建 Milvus 客户端：
milvus_client = ops.ann_search.milvus_client(host='192.168.68.108', 
                                             port='19530',
                                             collection_name='search_article_in_medium',
                                             output_fields=['title'])

@timer
def search():
    search_pipe = (pipe.input('query')
                    .map('query', 'vec', dpr_model)
                    .map('vec', 'vec', lambda x: x / np.linalg.norm(x, axis=0))
                    .flat_map('vec', ('id', 'score', 'title'), milvus_client)  
                    .output('query', 'id', 'score', 'title'))

    res = search_pipe('Building high performance startup teams')
    DataCollection(res).show()
    print('ok')


if __name__ == '__main__':
    search()