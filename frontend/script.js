// =========================================================================================
// STATE APLIKASI
// =========================================================================================
let currentSessionId = null;
let currentStage = 1;
let currentObject = null; // Objek Three.js yang sedang ditampilkan

// =========================================================================================
// KONFIGURASI & INISIALISASI THREE.JS
// =========================================================================================
let scene, camera, renderer, controls;
const blockAtlas = {
    texture: null,
    uvMap: null,
    atlasSize: 0,
};

function initViewer() {
    const container = document.getElementById('viewer');
    if (!container) return;

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f4f8);

    // Camera
    camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.set(0, 1, 3);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);

    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 10, 7);
    scene.add(directionalLight);

    // Grid
    const gridHelper = new THREE.GridHelper(10, 10);
    scene.add(gridHelper);

    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    animate();
}

// Menangani resize window
window.addEventListener('resize', () => {
    if (!renderer || !camera) return;
    const container = document.getElementById('viewer');
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
});


// =========================================================================================
// FUNGSI-FUNGSI UI (User Interface)
// =========================================================================================

function showStatus(message) {
    document.getElementById('status-text').textContent = message;
    document.getElementById('status-overlay').style.display = 'flex';
}

function hideStatus() {
    document.getElementById('status-overlay').style.display = 'none';
}

function showError(message) {
    document.getElementById('error-text').textContent = message;
    document.getElementById('error-overlay').style.display = 'flex';
    hideStatus(); // Sembunyikan status loading jika ada
}

function hideError() {
    document.getElementById('error-overlay').style.display = 'none';
}

function clearViewer() {
    if (currentObject) {
        scene.remove(currentObject);
        currentObject = null;
    }
    // Hapus juga geometri dan material untuk membebaskan memori
    // (Implementasi lebih lanjut bisa ditambahkan di sini jika perlu)
}

function resetToStage(stage) {
    currentStage = stage;
    // Sembunyikan semua div tahapan
    document.getElementById('stage1Div').style.display = 'none';
    document.getElementById('stage2Div').style.display = 'none';
    document.getElementById('stage3Div').style.display = 'none';
    document.getElementById('stage4Div').style.display = 'none';
    document.getElementById('resetDiv').style.display = 'none';

    // Tampilkan div untuk tahap saat ini
    if (stage === 1) {
        document.getElementById('stage1Div').style.display = 'block';
        document.getElementById('reconstructForm').reset();
        const preview = document.getElementById('imagePreview');
        preview.style.display = 'none';
        preview.src = '';
        currentSessionId = null;
        clearViewer();
    } else {
        document.getElementById(`stage${stage}Div`).style.display = 'block';
        document.getElementById('resetDiv').style.display = 'block';
    }
}

// =========================================================================================
// LOGIKA UTAMA APLIKASI
// =========================================================================================

const API_BASE_URL = 'http://localhost:8000'; // Sesuaikan jika backend Anda di host berbeda

// Tahap 1: Rekonstruksi
async function handleReconstruction(event) {
    event.preventDefault();
    const form = event.target;
    const formData = new FormData(form);
    
    showStatus('Tahap 1/4: Mengunggah gambar...');
    
    try {
        const response = await fetch(`${API_BASE_URL}/reconstruct`, {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || 'Gagal melakukan rekonstruksi.');
        }
        
        showStatus('Tahap 1/4: Memuat model 3D...');
        currentSessionId = data.sessionId;
        
        const objLoader = new THREE.OBJLoader();
        const textureLoader = new THREE.TextureLoader();

        const texture = await textureLoader.loadAsync(API_BASE_URL + data.textureUrl);
        const material = new THREE.MeshStandardMaterial({ map: texture });
        
        const object = await objLoader.loadAsync(API_BASE_URL + data.objUrl);
        object.traverse(child => {
            if (child instanceof THREE.Mesh) {
                child.material = material;
            }
        });
        
        clearViewer();
        scene.add(object);
        currentObject = object;
        
        hideStatus();
        resetToStage(2);

    } catch (error) {
        console.error('Error in reconstruction:', error);
        showError(`Kesalahan pada Tahap 1: ${error.message}`);
    }
}

// Tahap 2: Voxelization
async function handleVoxelization(event) {
    event.preventDefault();
    if (!currentSessionId) {
        showError("Session ID tidak ditemukan. Harap mulai dari awal.");
        return;
    }

    const payload = {
        sessionId: currentSessionId,
        maxBlocks: parseInt(document.getElementById('maxBlocks').value, 10),
        fill: document.getElementById('fillModel').checked,
    };

    showStatus('Tahap 2/4: Proses Voxelization...');

    try {
        const response = await fetch(`${API_BASE_URL}/voxelize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || 'Gagal melakukan voxelization.');
        }

        // Visualisasikan voxels
        clearViewer();
        const voxelGroup = new THREE.Group();
        const geometry = new THREE.BoxGeometry(1, 1, 1);
        
        for (const key in data.voxels) {
            const voxel = data.voxels[key];
            const pos = key.split(',').map(Number);
            
            const material = new THREE.MeshStandardMaterial({ 
                color: new THREE.Color(`rgb(${voxel.r}, ${voxel.g}, ${voxel.b})`)
            });
            const cube = new THREE.Mesh(geometry, material);
            cube.position.set(pos[0], pos[1], pos[2]);
            voxelGroup.add(cube);
        }

        // Center the voxel group
        const box = new THREE.Box3().setFromObject(voxelGroup);
        const center = box.getCenter(new THREE.Vector3());
        voxelGroup.position.sub(center);

        scene.add(voxelGroup);
        currentObject = voxelGroup;

        hideStatus();
        resetToStage(3);

    } catch (error) {
        console.error('Error in voxelization:', error);
        showError(`Kesalahan pada Tahap 2: ${error.message}`);
    }
}

// Tahap 3: Pemetaan Blok
async function handleBlockMapping() {
    if (!currentSessionId) {
        showError("Session ID tidak ditemukan. Harap mulai dari awal.");
        return;
    }
    
    showStatus('Tahap 3/4: Memetakan ke blok Minecraft...');

    try {
        const response = await fetch(`${API_BASE_URL}/map-blocks`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sessionId: currentSessionId }),
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || 'Gagal memetakan blok.');
        }

        // Visualisasikan blok
        await visualizeBlocks(data.blocks);
        
        hideStatus();
        resetToStage(4);

    } catch (error) {
        console.error('Error in block mapping:', error);
        showError(`Kesalahan pada Tahap 3: ${error.message}`);
    }
}

// Tahap 4: Ekspor
async function handleExport(format) {
     if (!currentSessionId) {
        showError("Session ID tidak ditemukan. Harap mulai dari awal.");
        return;
    }

    showStatus(`Tahap 4/4: Mengekspor file .${format}...`);

    try {
        const response = await fetch(`${API_BASE_URL}/export-${format}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sessionId: currentSessionId }),
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || `Gagal mengekspor .${format}.`);
        }
        
        // Memicu unduhan
        window.location.href = API_BASE_URL + data.downloadUrl;

        hideStatus();

    } catch (error) {
        console.error(`Error exporting .${format}:`, error);
        showError(`Kesalahan pada Tahap 4: ${error.message}`);
    }
}


// =========================================================================================
// FUNGSI BANTU & EVENT LISTENERS
// =========================================================================================

async function loadBlockAtlasData() {
    try {
        const response = await fetch('./assets/vanilla.atlas');
        const data = await response.json();
        blockAtlas.uvMap = {};
        data.blocks.forEach(b => {
            blockAtlas.uvMap[b.name] = b.faces; // Sederhanakan, kita ambil semua muka
        });
        
        // Untuk visualisasi, kita perlu atlas texture dan info UV yang lebih sederhana
        // Ini adalah data yang diproses sebelumnya, idealnya dihasilkan oleh skrip terpisah
        const processedAtlasResponse = await fetch('./assets/processed_atlas.json');
        const processedData = await processedAtlasResponse.json();
        blockAtlas.uvMap = processedData.uvMap;
        blockAtlas.atlasSize = processedData.atlasSize;
        blockAtlas.texture = await new THREE.TextureLoader().loadAsync('./assets/vanilla.png');
        blockAtlas.texture.magFilter = THREE.NearestFilter;
        blockAtlas.texture.minFilter = THREE.NearestFilter;

    } catch (error) {
        console.error("Gagal memuat data atlas blok:", error);
        showError("Tidak dapat memuat aset game. Beberapa visualisasi mungkin tidak berfungsi.");
    }
}

async function visualizeBlocks(blocks) {
    if (!blockAtlas.texture || !blockAtlas.uvMap) {
        showError("Data Atlas Blok tidak dimuat, tidak dapat memvisualisasikan.");
        return;
    }

    clearViewer();
    const finalGroup = new THREE.Group();
    const meshesByMaterial = {};

    // Kelompokkan geometri berdasarkan nama blok (material)
    blocks.forEach(block => {
        const blockName = block.name || "minecraft:stone";
        if (!meshesByMaterial[blockName]) {
            meshesByMaterial[blockName] = [];
        }
        const position = block.position;
        const matrix = new THREE.Matrix4().makeTranslation(position.x, position.y, position.z);
        meshesByMaterial[blockName].push(matrix);
    });

    const baseGeometry = new THREE.BoxGeometry(1, 1, 1);

    for (const blockName in meshesByMaterial) {
        const matrices = meshesByMaterial[blockName];
        const instancedMesh = new THREE.InstancedMesh(baseGeometry, new THREE.MeshBasicMaterial(), matrices.length);
        
        for (let i = 0; i < matrices.length; i++) {
            instancedMesh.setMatrixAt(i, matrices[i]);
        }

        // Buat material unik untuk setiap jenis blok
        const uvData = blockAtlas.uvMap[blockName] || blockAtlas.uvMap['minecraft:stone'];
        const tileX = uvData.x;
        const tileY = uvData.y;
        const textureSize = 1.0 / blockAtlas.atlasSize;
        const u0 = tileX * textureSize;
        const v0 = 1.0 - (tileY + 1) * textureSize;
        
        const texture = blockAtlas.texture.clone();
        texture.needsUpdate = true;
        texture.offset.set(u0, v0);
        texture.repeat.set(textureSize, textureSize);

        instancedMesh.material = new THREE.MeshStandardMaterial({ map: texture });
        instancedMesh.instanceMatrix.needsUpdate = true;
        finalGroup.add(instancedMesh);
    }

    const box = new THREE.Box3().setFromObject(finalGroup);
    const center = box.getCenter(new THREE.Vector3());
    finalGroup.position.sub(center);

    scene.add(finalGroup);
    currentObject = finalGroup;
}


function setupEventListeners() {
    // Form submissions
    document.getElementById('reconstructForm').addEventListener('submit', handleReconstruction);
    document.getElementById('voxelizeForm').addEventListener('submit', handleVoxelization);

    // Button clicks
    document.getElementById('mapBtn').addEventListener('click', handleBlockMapping);
    document.getElementById('exportLitematicBtn').addEventListener('click', () => handleExport('litematic'));
    document.getElementById('exportSchemBtn').addEventListener('click', () => handleExport('schem'));
    document.getElementById('resetBtn').addEventListener('click', () => resetToStage(1));
    document.getElementById('close-error-btn').addEventListener('click', hideError);

    // Input file change
    const imageInput = document.getElementById('imageInput');
    const imagePreview = document.getElementById('imagePreview');
    const uploadBox = document.getElementById('uploadBox');
    
    uploadBox.addEventListener('click', () => imageInput.click());

    imageInput.addEventListener('change', () => {
        const file = imageInput.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    });

    // Slider value displays
    document.getElementById('resolution').addEventListener('input', e => {
        document.getElementById('resolutionValue').textContent = e.target.value;
    });
    document.getElementById('maxBlocks').addEventListener('input', e => {
        document.getElementById('maxBlocksValue').textContent = e.target.value;
    });
}

// =========================================================================================
// INISIALISASI APLIKASI
// =========================================================================================
async function init() {
    initViewer();
    setupEventListeners();
    await loadBlockAtlasData(); // Muat data atlas di awal
    resetToStage(1);
}

// Jalankan aplikasi setelah DOM dimuat
document.addEventListener('DOMContentLoaded', init);
