import logging
import numpy as np
import nbtlib
from nbtlib.tag import *
from block_mapper import BlockMesh 

# =========================================================================================
# FUNSI BANTU: ENCODE VARINT
# =========================================================================================

def encode_as_varint(palette_ids):
    """
    Meng-encode iterable integer menjadi stream VarInt (list of ints 0..255).
    Mengembalikan list of bytes (0..255).
    """
    encoded_bytes = []
    for pid in palette_ids:
        num = int(pid)
        while True:
            byte_part = num & 0x7F 
            num >>= 7
            if num != 0:
                byte_part |= 0x80 
            encoded_bytes.append(byte_part)
            if num == 0:
                break
    return encoded_bytes

def to_signed_byte_list(byte_list):
    """
    Konversi list nilai 0..255 ke -128..127 sesuai tipe ByteArray di NBT.
    """
    return [b if b < 128 else b - 256 for b in byte_list]

# =========================================================================================
# KELAS EXPORTER UTAMA
# =========================================================================================

class Exporter:
    def __init__(self, block_mesh: BlockMesh):
        """
        block_mesh: instance BlockMesh yang sudah terisi blok-blok Minecraft
        """
        self.block_mesh = block_mesh
        min_b, max_b = self.block_mesh.get_bounds()

        self.width = max(1, int(max_b.x - min_b.x + 1))   # X
        self.height = max(1, int(max_b.y - min_b.y + 1))  # Y
        self.length = max(1, int(max_b.z - min_b.z + 1))  # Z
        self.min_bounds = min_b

        self._build_palette()

    def _build_palette(self):
        """
        Bangun palette berdasarkan urutan kemunculan blok di mesh.
        Ini menghindari mismatch urutan.
        """
        self.unique_blocks = []
        self.palette_map = {}

        for block in self.block_mesh.get_blocks():
            if block.name not in self.palette_map:
                idx = len(self.unique_blocks)
                self.unique_blocks.append(block.name)
                self.palette_map[block.name] = idx

        # Pastikan ada air (minecraft:air)
        if "minecraft:air" not in self.palette_map:
            air_id = len(self.unique_blocks)
            self.unique_blocks.append("minecraft:air")
            self.palette_map["minecraft:air"] = air_id

    def export_to_schem_v2(self, filename: str, data_version: int = 3953):
        """
        Mengekspor struktur ke .schem (Schematic v2 - Sponge/WorldEdit compatible).
        filename: lokasi output .schem (NBT gzipped)
        data_version: Minecraft data version (default 3953 untuk MC 1.20.1)
        """
        logging.info(f"Mengekspor ke format .schem v2: {filename}")

        # --- Build NBT Palette ---
        nbt_palette = Compound({name: Int(i) for name, i in self.palette_map.items()})

        # --- Buat array 3D (Y, Z, X) ---
        block_ids = np.full(
            (self.height, self.length, self.width),
            self.palette_map["minecraft:air"],
            dtype=np.int32
        )

        for block in self.block_mesh.get_blocks():
            x = int(block.position.x - self.min_bounds.x)
            y = int(block.position.y - self.min_bounds.y)
            z = int(block.position.z - self.min_bounds.z)
            if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.length:
                block_ids[y, z, x] = self.palette_map.get(block.name, self.palette_map["minecraft:air"])

        # Flatten ke urutan YZX (C-order)
        flat_ids = block_ids.flatten()

        # --- Encode ke VarInt ---
        varint_bytes = encode_as_varint(flat_ids)
        signed_varint_bytes = to_signed_byte_list(varint_bytes)

        # --- Struktur NBT v2 ---
        schem_root = {
            'Version': Int(2),
            'DataVersion': Int(data_version),
            'Width': Short(self.width),
            'Height': Short(self.height),
            'Length': Short(self.length),
            'PaletteMax': Int(len(self.palette_map)),
            'Palette': nbt_palette,
            'BlockData': ByteArray(signed_varint_bytes),
            'Entities': List[Compound]([]),
            'BlockEntities': List[Compound]([]),
            'Offset': List[Int]([Int(0), Int(0), Int(0)]),
        }

        schem_nbt = nbtlib.File(schem_root, gzipped=True)
        schem_nbt.save(filename)

        logging.info("Ekspor .schem v2 selesai.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
