import os
import uuid
import json
from pathlib import Path
import numpy as np
import logging
import imghdr
import shutil
import asyncio
from datetime import datetime, timedelta

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Impor modul-modul proyek
import trimesh
from PIL import Image
from triposr_runner import run_triposr
from core_voxelizer import BasicGridVoxeliser, VoxelMesh
from block_mapper import map_voxels_to_blocks, load_atlas_data, calculate_face_visibility, BlockMesh
from exporter import Exporter

# === KONFIGURASI APLIKASI ===
TEMP_DIR = "temp"
ASSET_DIR = Path("frontend") / "assets"
# PERBAIKAN: Menggunakan file atlas yang sesuai untuk pemetaan warna
ATLAS_PATH = ASSET_DIR / "vanilla.atlas"
ALLOWED_IMAGE_EXT = ['jpeg', 'png', 'jpg', 'bmp']
CORS_ALLOW = ["*"]
MAX_FILE_SIZE_MB = 10
SESSION_LIFESPAN_HOURS = 24

SESSION_STORAGE = {}

app = FastAPI(title="3D to Schematic API")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ALLOW, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

os.makedirs(TEMP_DIR, exist_ok=True)
app.mount("/temp", StaticFiles(directory=TEMP_DIR), name="temp_files")
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s %(message)s')

try:
    ATLAS_DATA = load_atlas_data(str(ATLAS_PATH))
except Exception as e:
    logging.fatal(f"KRITIS: Gagal memuat file atlas '{ATLAS_PATH}'. Error: {e}")
    exit(1)
    
scheduler = AsyncIOScheduler()

def cleanup_old_sessions():
    now = datetime.now()
    lifespan = timedelta(hours=SESSION_LIFESPAN_HOURS)
    temp_path = Path(TEMP_DIR)
    
    logging.info("Menjalankan tugas pembersihan sesi...")
    cleaned_count = 0
    for session_dir in temp_path.iterdir():
        if session_dir.is_dir():
            try:
                dir_time = datetime.fromtimestamp(session_dir.stat().st_mtime)
                if now - dir_time > lifespan:
                    if session_dir.name in SESSION_STORAGE:
                        del SESSION_STORAGE[session_dir.name]
                    shutil.rmtree(session_dir)
                    logging.info(f"Menghapus sesi usang: {session_dir.name}")
                    cleaned_count += 1
            except Exception as e:
                logging.error(f"Gagal menghapus direktori {session_dir.name}: {e}")
    if cleaned_count > 0:
        logging.info(f"Pembersihan selesai. {cleaned_count} sesi dihapus.")

@app.on_event("startup")
def start_scheduler(): 
    scheduler.add_job(cleanup_old_sessions, 'interval', hours=6)
    scheduler.start()

@app.on_event("shutdown")
def shutdown_scheduler(): 
    scheduler.shutdown()

class VoxelizePayload(BaseModel):
    sessionId: str
    max_blocks: int = 128
    fill: bool = True

class MapPayload(BaseModel):
    sessionId: str

class ExportPayload(BaseModel):
    sessionId: str

def secure_filename(filename: str) -> str: 
    return Path(filename).name.replace("..", "").replace("/", "").replace("\\", "")

def validate_image(file: UploadFile):
    if file.size > MAX_FILE_SIZE_MB * 1024 * 1024: 
        raise HTTPException(status_code=413, detail=f"Ukuran file melebihi {MAX_FILE_SIZE_MB}MB.")
    
    contents = file.file.read()
    file.file.seek(0)
    
    kind = imghdr.what(None, contents)
    if kind not in ALLOWED_IMAGE_EXT: 
        raise HTTPException(status_code=400, detail="File bukan gambar yang valid. Hanya menerima JPEG, PNG, BMP.")

@app.post("/reconstruct", summary="Tahap 1: Rekonstruksi Gambar ke 3D")
async def reconstruct(image: UploadFile = File(...), remove_bg: bool = Form(True), resolution: int = Form(256)):
    validate_image(image)
    session_id = str(uuid.uuid4())
    session_dir = Path(TEMP_DIR) / session_id
    os.makedirs(session_dir, exist_ok=True)
    
    safe_filename = secure_filename(image.filename)
    input_img_path = session_dir / safe_filename
    
    with open(input_img_path, "wb") as f: 
        shutil.copyfileobj(image.file, f)

    try:
        logging.info(f"[{session_id}] Memulai rekonstruksi 3D...")
        obj_path, texture_path, _ = await asyncio.to_thread(
            run_triposr, str(input_img_path), str(session_dir), resolution=resolution, remove_bg=remove_bg
        )
        
        def get_url(p): 
            return f"/temp/{session_id}/{Path(p).name}" if p and os.path.exists(p) else None
            
        return JSONResponse({
            "sessionId": session_id, 
            "objUrl": get_url(obj_path), 
            "textureUrl": get_url(texture_path)
        })
    except Exception as e:
        logging.error(f"[{session_id}] Rekonstruksi gagal: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Kesalahan internal saat rekonstruksi model 3D.")

@app.post("/voxelize", summary="Tahap 2: Vokselisasi Model 3D")
async def voxelize(payload: VoxelizePayload):
    sessionId = secure_filename(payload.sessionId)
    session_dir = Path(TEMP_DIR) / sessionId
    obj_path = session_dir / "model.obj"
    texture_path = session_dir / "baked_texture.png"

    if not obj_path.exists() or not texture_path.exists():
        raise HTTPException(status_code=404, detail="File model atau tekstur tidak ditemukan untuk sesi ini.")
        
    try:
        logging.info(f"[{sessionId}] Memulai vokselisasi...")
        mesh = await asyncio.to_thread(trimesh.load, obj_path, force='mesh')
        texture = await asyncio.to_thread(Image.open, texture_path)
        
        voxelizer = BasicGridVoxeliser()
        voxel_mesh_obj, solid_grid = await asyncio.to_thread(
            voxelizer.run, mesh, texture, payload.max_blocks, payload.fill
        )
        
        SESSION_STORAGE[sessionId] = {'voxel_mesh': voxel_mesh_obj, 'solid_grid': solid_grid}
        
        preview_array = await asyncio.to_thread(voxel_mesh_obj.to_numpy_array)
        preview_path = session_dir / "voxel_preview.json"
        with open(preview_path, "w") as f: 
            json.dump(preview_array.tolist(), f)

        return JSONResponse({"sessionId": sessionId, "voxelPreviewUrl": f"/temp/{sessionId}/voxel_preview.json"})
    except Exception as e:
        logging.error(f"[{sessionId}] Vokselisasi gagal: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Kesalahan internal saat vokselisasi.")

@app.post("/map-blocks", summary="Tahap 3: Pemetaan Voxel ke Blok")
async def map_blocks(payload: MapPayload):
    sessionId = secure_filename(payload.sessionId)
    session_dir = Path(TEMP_DIR) / sessionId
    session_data = SESSION_STORAGE.get(sessionId)
    
    if not session_data or 'voxel_mesh' not in session_data or 'solid_grid' not in session_data:
        raise HTTPException(status_code=404, detail="Data vokselisasi tidak ditemukan. Jalankan tahap 2 dahulu.")
    
    try:
        voxel_mesh_obj = session_data['voxel_mesh']
        solid_grid = session_data['solid_grid']
        
        visibility_grid = await asyncio.to_thread(calculate_face_visibility, solid_grid)
        
        logging.info(f"[{sessionId}] Memulai pemetaan blok canggih...")
        block_mesh_obj = await asyncio.to_thread(map_voxels_to_blocks, voxel_mesh_obj, visibility_grid, ATLAS_DATA)

        session_data['block_mesh'] = block_mesh_obj
        
        block_name_grid = np.full(solid_grid.shape, "minecraft:air", dtype=object)
        for block in block_mesh_obj.get_blocks():
            x, y, z = int(block.position.x), int(block.position.y), int(block.position.z)
            if 0 <= x < block_name_grid.shape[0] and 0 <= y < block_name_grid.shape[1] and 0 <= z < block_name_grid.shape[2]:
                block_name_grid[x, y, z] = block.name
            
        block_names_path = session_dir / "block_names_preview.json"
        with open(block_names_path, "w") as f:
            json.dump(block_name_grid.tolist(), f)

        return JSONResponse({"sessionId": sessionId, "blockNamesPreviewUrl": f"/temp/{sessionId}/block_names_preview.json"})
    except Exception as e:
        logging.error(f"[{sessionId}] Pemetaan blok gagal: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Kesalahan internal saat pemetaan blok.")

# PERBAIKAN: Fungsi internal untuk menjalankan ekspor
def _run_export(block_mesh: BlockMesh, session_dir: Path, format_type: str):
    if not block_mesh:
        raise ValueError("Data blok tidak ditemukan di sesi ini.")
    
    exporter = Exporter(block_mesh)
    
    if format_type == 'schem':
        output_path = session_dir / "output.schem"
        # PERBAIKAN UTAMA: Memanggil fungsi `export_to_schem_v2` yang benar
        exporter.export_to_schem_v2(str(output_path))
    # PERBAIKAN: Litematic dinonaktifkan karena belum ada implementasinya
    # elif format_type == 'litematic':
    #     output_path = session_dir / "output.litematic"
    #     # Anda perlu menambahkan fungsi export_to_litematic di exporter.py
    #     exporter.export_to_litematic(str(output_path))
    else:
        raise ValueError("Format ekspor tidak valid.")
        
    return output_path

# PERBAIKAN: Menggabungkan logika export ke satu fungsi
async def _handle_export(payload: ExportPayload, format_type: str):
    sessionId = secure_filename(payload.sessionId)
    session_dir = Path(TEMP_DIR) / sessionId
    block_mesh = SESSION_STORAGE.get(sessionId, {}).get('block_mesh')

    if not block_mesh:
        raise HTTPException(status_code=404, detail="Data pemetaan blok tidak ditemukan. Jalankan tahap 3 dahulu.")
        
    try:
        output_path = await asyncio.to_thread(_run_export, block_mesh, session_dir, format_type)
        return JSONResponse({"downloadUrl": f"/temp/{sessionId}/{output_path.name}"})
    except Exception as e:
        logging.error(f"[{sessionId}] Ekspor .{format_type} gagal: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export-schematic", summary="Tahap 4: Ekspor ke .schem")
async def export_schematic(payload: ExportPayload):
    return await _handle_export(payload, 'schem')
        
# PERBAIKAN: Menonaktifkan endpoint litematic
# @app.post("/export-litematic", summary="Tahap 4: Ekspor ke .litematic")
# async def export_litematic(payload: ExportPayload):
#     return await _handle_export(payload, 'litematic')

# Sajikan frontend sebagai fallback
frontend_dir = Path(__file__).parent / "frontend"
if not frontend_dir.exists():
    frontend_dir = Path.cwd() / "frontend"

if frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
    logging.info(f"Menyajikan file frontend dari: {frontend_dir.resolve()}")
else:
    logging.warning(f"Direktori 'frontend' tidak ditemukan di {frontend_dir.resolve()}. Antarmuka web tidak akan tersedia.")