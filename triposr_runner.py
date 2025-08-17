import os
import numpy as np
import torch
import trimesh
from PIL import Image
import logging

# Pastikan rembg diimpor
import rembg

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video
from tsr.bake_texture import bake_texture as bake_texture_fn

def write_obj(vertices, uvs, faces, obj_path, texture_name="texture.png"):
    # Fungsi ini tidak perlu diubah
    with open(obj_path, 'w') as f:
        f.write(f"mtllib material.mtl\nusemtl material_0\n")
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for vt in uvs:
            f.write(f"vt {vt[0]} {1 - vt[1]}\n")  # OBJ UV Y-axis is inverted!
        for face in faces:
            idx = [i + 1 for i in face]
            f.write(f"f {idx[0]}/{idx[0]} {idx[1]}/{idx[1]} {idx[2]}/{idx[2]}\n")

    mtl_path = os.path.join(os.path.dirname(obj_path), "material.mtl")
    with open(mtl_path, "w") as f:
        f.write(f"newmtl material_0\n")
        f.write(f"map_Kd {texture_name}\n")

def run_triposr(
        image_path,
        output_dir,
        bake_texture=True,
        render=False,
        resolution=256,
        texture_resolution=2048,
        device=None,
        foreground_ratio=0.85,
        remove_bg=True,
        export_formats=["obj"]
):
    os.makedirs(output_dir, exist_ok=True)
    if device is None:
        device = os.environ.get("TRIPOSR_DEVICE", "cuda:0")
    if not torch.cuda.is_available():
        device = "cpu"
    logging.info(f"Menggunakan device: {device}")

    model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
    model.to(device)
    model.renderer.set_chunk_size(8192)

    rembg_session = rembg.new_session()
    
    # --- PERBAIKAN LOGIKA PENANGANAN GAMBAR TRANSPARAN ---
    logging.info("Memproses gambar input...")
    img_pil = Image.open(image_path)

    if remove_bg:
        # Cek apakah gambar sudah memiliki transparansi (alpha channel)
        if img_pil.mode == 'RGBA':
            logging.info("Gambar input sudah memiliki alpha channel. Melewatkan rembg.")
            image = resize_foreground(img_pil, foreground_ratio)
        else:
            logging.info("Menghapus latar belakang dengan rembg...")
            image = remove_background(img_pil, rembg_session)
            image = resize_foreground(image, foreground_ratio)
        
        image_np = np.array(image).astype(np.float32) / 255.0
        # Komposit ke latar belakang abu-abu, menggunakan alpha channel yang ada
        image_rgb = image_np[:, :, :3] * image_np[:, :, 3:4] + (1 - image_np[:, :, 3:4]) * 0.5
        image = Image.fromarray((image_rgb * 255.0).astype(np.uint8))
    else:
        logging.info("Menggunakan gambar asli tanpa menghapus latar belakang.")
        image = img_pil.convert("RGB")
    # --- PERBAIKAN LOGIKA SELESAI ---

    input_path = os.path.join(output_dir, "input.png")
    image.save(input_path)

    logging.info("Memulai inferensi model TripoSR...")
    with torch.no_grad():
        scene_codes = model([image], device=device)

    if render:
        logging.info("Merender video pratinjau...")
        render_images = model.render(scene_codes, n_views=30, return_type="pil")
        save_video(render_images[0], os.path.join(output_dir, "render.mp4"), fps=30)

    logging.info("Mengekstrak mesh dari model...")
    meshes = model.extract_mesh(scene_codes, True, resolution=resolution)
    mesh = meshes[0]

    output_obj_path = os.path.join(output_dir, "model.obj")
    texture_path = os.path.join(output_dir, "baked_texture.png") if bake_texture else None
    extras = {}

    if bake_texture:
        logging.info("Mem-bake tekstur ke mesh...")
        baked = bake_texture_fn(mesh, model, scene_codes[0], texture_resolution, output_dir=output_dir)

        vertices = mesh.vertices[baked["vmapping"]]
        uvs = baked["uvs"]
        faces = baked["indices"]

        R = trimesh.transformations.rotation_matrix(
            np.radians(-90), [1, 0, 0], point=[0, 0, 0]
        )
        vertices = trimesh.transform_points(vertices, R)

        texture_path = baked["texture_path"]
        write_obj(vertices, uvs, faces, output_obj_path, os.path.basename(texture_path))
        logging.info(f"Model OBJ disimpan di: {output_obj_path}")
        mesh = None
    else:
        mesh.export(output_obj_path)

    extras["obj"] = output_obj_path
    extras["texture"] = texture_path
    extras["mtl"] = os.path.join(output_dir, "material.mtl")

    return output_obj_path, texture_path, extras