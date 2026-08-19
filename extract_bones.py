import json
import sys

def get_nodes(filepath):
    try:
        with open(filepath, 'rb') as f:
            f.read(12) # skip header
            chunk_len = int.from_bytes(f.read(4), 'little')
            f.read(4) # skip type (JSON)
            data = json.loads(f.read(chunk_len).decode('utf-8'))
            nodes = [n.get('name') for n in data.get('nodes', []) if 'name' in n]
            print(nodes)
    except Exception as e:
        print("Error:", e)

get_nodes('static/model/Man_Mesh.glb')
