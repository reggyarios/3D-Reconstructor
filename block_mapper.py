import json
import numpy as np
import logging
from dataclasses import dataclass
from enum import Flag, auto

# Pastikan kelas-kelas data dari core_voxelizer diimpor dengan benar
from core_voxelizer import VoxelMesh, Vector3, RGBA

# Kelas untuk menampung hasil pemetaan blok, tidak ada perubahan
@dataclass
class Block:
    position: Vector3
    name: str
    colour: RGBA

# Kelas untuk menyimpan kumpulan blok hasil pemetaan, tidak ada perubahan
class BlockMesh:
    def __init__(self, voxel_mesh: VoxelMesh):
        self._blocks: list[Block] = []
        self._voxel_mesh = voxel_mesh
        self._palette: set[str] = set()

    def add_block(self, block: Block):
        self._blocks.append(block)
        self._palette.add(block.name)

    def get_blocks(self) -> list[Block]:
        return self._blocks
    
    def get_block_palette(self) -> list[str]:
        return sorted(list(self._palette))

    def get_bounds(self):
        return self._voxel_mesh.get_bounds()

# Enum untuk visibilitas sisi, tidak ada perubahan
class FaceVisibility(Flag):
    NONE = 0
    UP = auto()
    DOWN = auto()
    NORTH = auto()
    EAST = auto()
    SOUTH = auto()
    WEST = auto()

# Kelas untuk data atlas, tidak ada perubahan
@dataclass
class FaceData:
    colour: RGBA
    std: float 

@dataclass
class AtlasBlock:
    name: str
    colour: RGBA 
    faces: dict[str, FaceData]

def load_atlas_data(atlas_path: str) -> dict[str, AtlasBlock]:
    """
    Memuat dan mem-parsing file .atlas.
    Fungsi ini sudah robust dan tidak perlu diubah.
    """
    atlas_data = {}
    try:
        with open(atlas_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Iterasi melalui 'blocks' untuk mendapatkan nama dan warna rata-rata
            for block_data in data['blocks']:
                name = block_data['name']
                col = block_data['colour']
                avg_colour = RGBA(int(col['r']*255), int(col['g']*255), int(col['b']*255))
                
                faces = {}
                # Iterasi melalui 'faces' untuk menautkan ke data tekstur spesifik
                for face_name, face_info_key in block_data['faces'].items():
                    if face_info_key in data['textures']:
                        texture_info = data['textures'][face_info_key]
                        tex_col = texture_info['colour']
                        faces[face_name] = FaceData(
                            colour=RGBA(int(tex_col['r']*255), int(tex_col['g']*255), int(tex_col['b']*255)),
                            std=texture_info['std']
                        )
                atlas_data[name] = AtlasBlock(name=name, colour=avg_colour, faces=faces)
        logging.info(f"Atlas dengan {len(atlas_data)} blok berhasil dimuat.")
        return atlas_data
    except Exception as e:
        logging.error(f"Gagal memuat file atlas {atlas_path}: {e}")
        raise

def get_contextual_face_average(block: AtlasBlock, visibility: FaceVisibility) -> RGBA:
    """
    Menghitung warna rata-rata dari sisi yang terlihat saja.
    Ini adalah inti dari pemilihan blok yang "pintar".
    """
    avg_r, avg_g, avg_b, count = 0, 0, 0, 0
    
    # Kumpulkan semua data wajah yang terlihat
    visible_faces_data = []
    if visibility & FaceVisibility.UP: visible_faces_data.append(block.faces.get('up'))
    if visibility & FaceVisibility.DOWN: visible_faces_data.append(block.faces.get('down'))
    if visibility & FaceVisibility.NORTH: visible_faces_data.append(block.faces.get('north'))
    if visibility & FaceVisibility.SOUTH: visible_faces_data.append(block.faces.get('south'))
    if visibility & FaceVisibility.EAST: visible_faces_data.append(block.faces.get('east'))
    if visibility & FaceVisibility.WEST: visible_faces_data.append(block.faces.get('west'))

    for face_data in visible_faces_data:
        if face_data:
            avg_r += face_data.colour.r
            avg_g += face_data.colour.g
            avg_b += face_data.colour.b
            count += 1

    # Jika tidak ada sisi yang terlihat (misal: blok di bagian dalam), gunakan warna rata-rata blok
    if count == 0:
        return block.colour

    return RGBA(int(avg_r/count), int(avg_g/count), int(avg_b/count))

def calculate_face_visibility(solid_grid: np.ndarray) -> np.ndarray:
    """
    Membuat grid yang berisi bitmask visibilitas untuk setiap voxel.
    Tidak ada perubahan, fungsi ini sudah optimal.
    """
    logging.info("Menghitung visibilitas sisi voxel...")
    visibility_grid = np.full(solid_grid.shape, FaceVisibility.NONE, dtype=object)
    
    # Padding untuk memudahkan pengecekan tetangga tanpa error out-of-bounds
    padded_grid = np.pad(solid_grid, 1, mode='constant', constant_values=False)
    
    indices = np.argwhere(solid_grid)
    for x, y, z in indices:
        px, py, pz = x + 1, y + 1, z + 1
        visibility = FaceVisibility.NONE
        if not padded_grid[px, py + 1, pz]: visibility |= FaceVisibility.UP
        if not padded_grid[px, py - 1, pz]: visibility |= FaceVisibility.DOWN
        if not padded_grid[px, py, pz - 1]: visibility |= FaceVisibility.NORTH
        if not padded_grid[px + 1, py, pz]: visibility |= FaceVisibility.EAST
        if not padded_grid[px, py, pz + 1]: visibility |= FaceVisibility.SOUTH
        if not padded_grid[px - 1, py, pz]: visibility |= FaceVisibility.WEST
        visibility_grid[x, y, z] = visibility
        
    return visibility_grid

def map_voxels_to_blocks(voxel_mesh: VoxelMesh, visibility_grid: np.ndarray, atlas: dict[str, AtlasBlock]) -> BlockMesh:
    """
    Fungsi utama yang memetakan setiap voxel ke blok Minecraft yang paling sesuai.
    Versi ini ditingkatkan dengan error handling yang lebih baik.
    """
    block_mesh_result = BlockMesh(voxel_mesh)
    cache = {} # Cache untuk mempercepat proses jika ada warna & visibilitas yang sama
    available_blocks = list(atlas.values())
    
    logging.info(f"Memetakan {voxel_mesh.get_voxel_count()} voxel ke palet blok...")

    for key, colour in voxel_mesh._voxels.items():
        try:
            coords = [int(c) for c in key.split(',')]
            
            # PERBAIKAN: Menambahkan pemeriksaan batas untuk mencegah error out-of-bounds
            if not (0 <= coords[0] < visibility_grid.shape[0] and \
                    0 <= coords[1] < visibility_grid.shape[1] and \
                    0 <= coords[2] < visibility_grid.shape[2]):
                logging.warning(f"Koordinat voxel {coords} di luar batas. Melewatkan.")
                continue

            visibility = visibility_grid[coords[0], coords[1], coords[2]]
            
            cache_key = (colour.r, colour.g, colour.b, visibility)
            
            if cache_key in cache:
                chosen_block_name = cache[cache_key]
            else:
                min_error = float('inf')
                chosen_block = None
                
                for atlas_block in available_blocks:
                    # Dapatkan warna kontekstual berdasarkan sisi yang terlihat
                    contextual_colour = get_contextual_face_average(atlas_block, visibility)
                    
                    # Hitung error (jarak kuadrat Euclidean di ruang warna RGB)
                    error = ( (colour.r - contextual_colour.r)**2 + 
                              (colour.g - contextual_colour.g)**2 + 
                              (colour.b - contextual_colour.b)**2 )
                    
                    if error < min_error:
                        min_error = error
                        chosen_block = atlas_block
                
                # Gunakan nama blok terpilih, atau 'stone' jika tidak ada yang cocok
                chosen_block_name = chosen_block.name if chosen_block else "minecraft:stone"
                cache[cache_key] = chosen_block_name

            block = Block(
                position=Vector3(coords[0], coords[1], coords[2]),
                name=chosen_block_name,
                colour=colour
            )
            block_mesh_result.add_block(block)

        except Exception as e:
            logging.error(f"Error saat memproses voxel di '{key}': {e}")
            continue # Lanjutkan ke voxel berikutnya jika terjadi error

    logging.info("Pemetaan blok selesai.")
    return block_mesh_result