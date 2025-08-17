# 3D Model Reconstruction 🏗️

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white "Python")](https://www.python.org/)  [![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white "NumPy")](https://numpy.org/)  [![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white "Flask")](https://flask.palletsprojects.com/) [![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black "JavaScript")](https://www.javascript.com/) [![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white "CSS3")](https://www.w3.org/Style/CSS/)

Proyek ini bertujuan untuk merekonstruksi model 3D dari input gambar yang diberikan, kemudian memvoxelisasi 3D model tersebut dengan output berformat ".scheme"

## Fitur Utama ✨
*   **Rekonstruksi 3D**: Menggunakan model TripoSR untuk merekonstruksi 3d model dari gambar.
*   **Voxelisasi Inti 🧱**: Memproses data input dan mengubahnya menjadi representasi voxel 3D.
*   **Pemetaan Blok 🗺️**: Mengelola dan memetakan voxel-voxel untuk membentuk model 3D yang lebih kompleks.
*   **Frontend Interaktif 🌐**: Antarmuka pengguna berbasis web (HTML, CSS, JavaScript) untuk berinteraksi dengan model yang direkonstruksi.
*   **Ekspor Model 📤**: Kemampuan untuk mengekspor model 3D yang direkonstruksi ke format scheme.

## Tech Stack 🛠️

*   Python 🐍
*   JavaScript ☕
*   HTML 🌐
*   CSS 🎨

## Instalasi & Menjalankan 🚀

1. Clone repositori:

   ```bash
   git clone https://github.com/reggyarios/3D-Reconstructor
   ```

2. Masuk ke direktori:

   ```bash
   cd 3D-Reconstructor
   ```

3. Install dependensi:

   ```bash
   pip install -r requirements.txt
   ```

4. Jalankan proyek:

   ```bash
   uvicorn main:app
   ```

## Cara Berkontribusi 🤝

1.  Fork repositori ini.
2.  Buat branch untuk fitur Anda (`git checkout -b feature/FiturBaru`).
3.  Lakukan commit perubahan Anda (`git commit -am 'Tambahkan beberapa fitur baru'`).
4.  Push ke branch (`git push origin feature/FiturBaru`).
5.  Buat Pull Request.

