import sys
import json
import numpy as np
from voxelizer import voxelize

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)

if len(sys.argv) != 5:
    print("Usage: voxelize_runner.py <input.obj> <output_dir> <max_blocks> <fill>")
    sys.exit(1)

obj_path = sys.argv[1]
out_dir = sys.argv[2]
max_blocks = int(sys.argv[3])
fill = bool(int(sys.argv[4]))

if not (16 <= max_blocks <= 512):
    print("max_blocks out of valid range (16-512)", file=sys.stderr)
    sys.exit(1)

print(f"[Runner] Voxelizing {obj_path} into {out_dir}/output_voxel.json")
try:
    grid = voxelize(obj_path, max_blocks, fill)
    output_grid = grid.tolist()
    with open(f"{out_dir}/output_voxel.json", "w") as f:
        json.dump(output_grid, f, cls=NumpyEncoder)
    print("[Runner] Selesai menulis output_voxel.json")
except Exception as e:
    print(f"Error selama voxelisasi: {e}", file=sys.stderr)
    sys.exit(1)
