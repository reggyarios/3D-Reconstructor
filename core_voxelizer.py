import numpy as np
import trimesh
from PIL import Image
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from scipy.spatial import KDTree

# =========================================================================================
# 1. KELAS-KELAS DATA
# =========================================================================================

@dataclass
class RGBA:
    r: int
    g: int
    b: int
    a: int = 255
    def to_list(self): return [self.r, self.g, self.b]

@dataclass
class Vector3:
    x: float
    y: float
    z: float
    def add(self, other: 'Vector3') -> 'Vector3': return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    def sub(self, other: 'Vector3') -> 'Vector3': return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    def to_array(self) -> np.ndarray: return np.array([self.x, self.y, self.z])

class VoxelMesh:
    def __init__(self):
        self._voxels: dict[str, RGBA] = {}
        self._min_bounds = Vector3(math.inf, math.inf, math.inf)
        self._max_bounds = Vector3(-math.inf, -math.inf, -math.inf)
        
    def _update_bounds(self, x, y, z):
        self._min_bounds.x = min(self._min_bounds.x, x)
        self._min_bounds.y = min(self._min_bounds.y, y)
        self._min_bounds.z = min(self._min_bounds.z, z)
        self._max_bounds.x = max(self._max_bounds.x, x)
        self._max_bounds.y = max(self._max_bounds.y, y)
        self._max_bounds.z = max(self._max_bounds.z, z)
        
    def add_voxel(self, x: int, y: int, z: int, colour: RGBA):
        key = f"{x},{y},{z}"
        self._voxels[key] = colour
        self._update_bounds(x, y, z)
        
    def get_voxel_count(self) -> int: 
        return len(self._voxels)
        
    def get_bounds(self) -> tuple[Vector3, Vector3]:
        if self.get_voxel_count() == 0: 
            return Vector3(0,0,0), Vector3(0,0,0)
        return self._min_bounds, self._max_bounds
        
    def to_numpy_array(self) -> np.ndarray:
        if self.get_voxel_count() == 0: 
            return np.zeros((1, 1, 1, 3), dtype=np.uint8)
            
        min_b, max_b = self.get_bounds()
        dims = (int(max_b.x - min_b.x + 1), int(max_b.y - min_b.y + 1), int(max_b.z - min_b.z + 1))
        grid_rgb = np.zeros(dims + (3,), dtype=np.uint8)
        
        for key, colour in self._voxels.items():
            coords = [int(c) for c in key.split(',')]
            local_x = coords[0] - int(min_b.x)
            local_y = coords[1] - int(min_b.y)
            local_z = coords[2] - int(min_b.z)
            grid_rgb[local_x, local_y, local_z] = colour.to_list()
            
        return grid_rgb

# =========================================================================================
# 2. KELAS VOXELISER ABSTRAK
# =========================================================================================

class Voxeliser(ABC):
    def run(self, mesh: trimesh.Trimesh, texture: Image.Image, max_blocks: int, fill: bool) -> tuple[VoxelMesh, np.ndarray]:
        voxel_mesh, solid_grid = self._voxelise(mesh, texture, max_blocks, fill)
        logging.info(f"Jumlah voxel yang dihasilkan: {voxel_mesh.get_voxel_count()}")
        min_b, max_b = voxel_mesh.get_bounds()
        dims = (int(max_b.x - min_b.x + 1), int(max_b.y - min_b.y + 1), int(max_b.z - min_b.z + 1))
        logging.info(f"Dimensi VoxelMesh: {dims[0]}x{dims[1]}x{dims[2]}")
        return voxel_mesh, solid_grid
    
    @abstractmethod
    def _voxelise(self, mesh: trimesh.Trimesh, texture: Image.Image, max_blocks: int, fill: bool) -> tuple[VoxelMesh, np.ndarray]:
        pass
    
    def _get_triangle_area(self, v0: Vector3, v1: Vector3, v2: Vector3) -> float:
        return 0.5 * np.linalg.norm(np.cross(v1.sub(v0).to_array(), v2.sub(v0).to_array()))

    def _get_voxel_colour(self, texture: Image.Image, face_verts: list, face_uvs: list, location_ws: Vector3) -> RGBA:
        v0, v1, v2 = [Vector3(*v) for v in face_verts]
        uv0, uv1, uv2 = face_uvs
        
        area01 = self._get_triangle_area(v0, v1, location_ws)
        area12 = self._get_triangle_area(v1, v2, location_ws)
        area20 = self._get_triangle_area(v2, v0, location_ws)
        total_area = area01 + area12 + area20
        
        if total_area < 1e-9: return RGBA(255, 0, 255)
        
        w0 = area12 / total_area
        w1 = area20 / total_area
        w2 = area01 / total_area
        
        final_u = uv0[0] * w0 + uv1[0] * w1 + uv2[0] * w2
        final_v = uv0[1] * w0 + uv1[1] * w1 + uv2[1] * w2
        
        if math.isnan(final_u) or math.isnan(final_v): return RGBA(255, 0, 255)
        
        tex_w, tex_h = texture.size
        tx = int(final_u * tex_w)
        ty = int((1 - final_v) * tex_h)
        
        tx = np.clip(tx, 0, tex_w - 1)
        ty = np.clip(ty, 0, tex_h - 1)
        
        pixel = texture.getpixel((tx, ty))
        if isinstance(pixel, int):
            return RGBA(pixel, pixel, pixel)
        elif len(pixel) == 3:
            return RGBA(pixel[0], pixel[1], pixel[2])
        elif len(pixel) == 4:
            return RGBA(pixel[0], pixel[1], pixel[2], pixel[3])
        return RGBA(255,0,255)

# =========================================================================================
# 3. IMPLEMENTASI KONKRET VOXELISER
# =========================================================================================

class BasicGridVoxeliser(Voxeliser):
    def _voxelise(self, mesh: trimesh.Trimesh, texture: Image.Image, max_blocks: int, fill: bool) -> tuple[VoxelMesh, np.ndarray]:
        voxel_mesh = VoxelMesh()
        
        min_bb, max_bb = mesh.bounds
        size = max_bb - min_bb
        
        if size.max() < 1e-6:
            return voxel_mesh, np.zeros((1,1,1), dtype=bool)
            
        pitch = size.max() / (max_blocks - 1) if max_blocks > 1 else size.max()
        if pitch < 1e-6: pitch = 1e-6
        
        logging.info("Membuat grid voxel solid menggunakan trimesh...")
        voxelized_grid = mesh.voxelized(pitch=pitch)
        solid_bool_grid = voxelized_grid.fill().matrix if fill else voxelized_grid.matrix

        from scipy.ndimage import binary_erosion
        surface_bool_grid = solid_bool_grid & ~binary_erosion(solid_bool_grid)
        
        surface_indices = np.argwhere(surface_bool_grid)
        if surface_indices.shape[0] == 0:
            logging.warning("Tidak ada voxel permukaan, menggunakan semua voxel solid.")
            surface_indices = np.argwhere(solid_bool_grid)

        if surface_indices.shape[0] == 0:
            logging.error("Vokselisasi gagal, tidak ada voxel yang ditemukan.")
            return voxel_mesh, solid_bool_grid

        logging.info(f"Mengidentifikasi {len(surface_indices)} voxel permukaan untuk diwarnai.")
        
        surface_world_coords = surface_indices * pitch + min_bb
        
        _, _, face_indices = trimesh.proximity.closest_point(mesh, surface_world_coords)

        for i, face_id in enumerate(face_indices):
            voxel_idx = surface_indices[i]
            location_ws = Vector3(*surface_world_coords[i])
            face_verts = mesh.vertices[mesh.faces[face_id]]
            face_uvs = mesh.visual.uv[mesh.faces[face_id]]
            
            colour = self._get_voxel_colour(texture, face_verts, face_uvs, location_ws)
            voxel_mesh.add_voxel(voxel_idx[0], voxel_idx[1], voxel_idx[2], colour)
        
        if fill:
            logging.info("Mengisi bagian dalam...")
            interior_mask = solid_bool_grid & ~surface_bool_grid
            interior_indices = np.argwhere(interior_mask)
            
            if interior_indices.shape[0] > 0 and surface_indices.shape[0] > 0:
                tree = KDTree(surface_indices)
                _, nearest_surface_indices = tree.query(interior_indices)
                
                for i, interior_idx in enumerate(interior_indices):
                    surface_idx = surface_indices[nearest_surface_indices[i]]
                    key = f"{surface_idx[0]},{surface_idx[1]},{surface_idx[2]}"
                    colour = voxel_mesh._voxels.get(key)
                    if colour:
                        voxel_mesh.add_voxel(interior_idx[0], interior_idx[1], interior_idx[2], colour)
        
        return voxel_mesh, solid_bool_grid
