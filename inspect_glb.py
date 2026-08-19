import json
import struct
import sys

def get_glb_nodes(filepath):
    with open(filepath, 'rb') as f:
        magic = f.read(4)
        if magic != b'glTF':
            print("Not a valid GLB file")
            return
            
        version = struct.unpack('<I', f.read(4))[0]
        length = struct.unpack('<I', f.read(4))[0]
        
        chunk0_length = struct.unpack('<I', f.read(4))[0]
        chunk0_type = f.read(4)
        
        if chunk0_type != b'JSON':
            print("First chunk is not JSON")
            return
            
        json_data = f.read(chunk0_length).decode('utf-8')
        gltf = json.loads(json_data)
        
        if 'nodes' in gltf:
            print("Found nodes:")
            for node in gltf['nodes']:
                if 'name' in node:
                    print(node['name'])
        else:
            print("No nodes found in GLTF.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        get_glb_nodes(sys.argv[1])
    else:
        print("Please provide a file path")
