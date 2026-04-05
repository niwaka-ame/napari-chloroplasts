import napari
import numpy as np
from pathlib import Path
from readlif.reader import LifFile
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QLabel,
    QFileDialog,
    QMessageBox,
    QLineEdit,
    QCheckBox,
    QDoubleSpinBox,
    QSpinBox,
    QGroupBox,
    QFormLayout,
)
from skimage import filters, morphology, exposure, measure, color, segmentation
import tifffile
from scipy.ndimage import binary_fill_holes
from porespy.filters import prune_branches
import omnipose
from omnipose.gpu import use_gpu
import cellpose_omni
from cellpose_omni import models
from cellpose_omni.models import MODEL_NAMES


# ==============================================================================
# --- SEGMENTATION ---
# ==============================================================================
def seg_wall(image_data, save_dir=None, filename_prefix="", gamma=0.5, otsu_mult=0.5):
    satos = []
    wall_img = exposure.adjust_gamma(image_data, gamma=gamma)
    for j in range(len(wall_img)):
        blurred = filters.gaussian(wall_img[j, :, :], sigma=1)
        edge_sato = filters.sato(blurred, black_ridges=False, sigmas=range(5, 14, 2))
        th_sato = filters.threshold_otsu(edge_sato)
        edge_sato = edge_sato >= th_sato * otsu_mult
        edge_sato = morphology.remove_small_objects(edge_sato, min_size=100)
        edge_sato = morphology.binary_dilation(edge_sato)
        edge_sato = morphology.skeletonize(edge_sato)
        edge_sato = prune_branches(edge_sato, iterations=5)
        satos.append(edge_sato)
        mask_3d = np.stack(satos, axis=0)
        mask_3d = mask_3d.astype(np.uint8) * 255
    if save_dir:
        # 1. Create the specific 'walls' subfolder inside your analysis folder
        wall_dir = Path(save_dir) / "walls"
        wall_dir.mkdir(parents=True, exist_ok=True)

        # 2. Construct the final file name
        file_path = wall_dir / f"{filename_prefix}_wall.tif"

        # 3. Write the 3D array to a multi-page TIFF
        tifffile.imwrite(file_path, mask_3d, imagej=True)
        print(f"Saved: {file_path}")
    return mask_3d


def seg_chlo(image_data, save_dir=None, filename_prefix="", use_gpu=False, niter=20):
    model_name = "nuclei"
    model = models.CellposeModel(gpu=use_gpu, model_type=model_name)
    # define parameters
    params = {
        "channels": None,  # always define this if using older models, e.g. [0,0] with bact_phase_omni
        "rescale": None,  # upscale or downscale your images, None = no rescaling
        "mask_threshold": 0,  # erode or dilate masks with higher or lower values between -5 and 5
        "flow_threshold": 0,  # default is .4, but only needed if there are spurious masks to clean up; slows down output
        "transparency": True,  # transparency in flow output
        "omni": True,  # we can turn off Omnipose mask reconstruction, not advised
        "cluster": True,  # use DBSCAN clustering
        "resample": True,  # whether or not to run dynamics on rescaled grid or original grid
        "verbose": False,  # turn on if you want to see more output
        "tile": False,  # average the outputs from flipped (augmented) images; slower, usually not needed
        "niter": 20,  # default None lets Omnipose calculate # of Euler iterations (usually <20) but you can tune it for over/under segmentation
        "augment": False,  # Can optionally rotate the image and average network outputs, usually not needed
        "affinity_seg": False,  # new feature, stay tuned...
    }
    all_masks = []
    for j in range(len(image_data)):
        masks, flows, styles = model.eval(image_data[j, :, :], **params)
        all_masks.append(masks)
    mask_3d = np.stack(all_masks, axis=0)
    if save_dir:
        # 1. Create the specific 'chlos' subfolder inside your analysis folder
        chlo_dir = Path(save_dir) / "chlos"
        chlo_dir.mkdir(parents=True, exist_ok=True)

        # 2. Construct the final file name
        file_path = chlo_dir / f"{filename_prefix}_chlo.tif"

        # 3. Write the 3D array to a multi-page TIFF
        tifffile.imwrite(file_path, mask_3d, imagej=True)
        print(f"Saved: {file_path}")
    return mask_3d


# ==============================================================================


class VeinViewerWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.lif_files = {}
        self.current_lif = None

        self.current_wall_data = None
        self.current_chlo_data = None

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # 1. Folder Selection Row
        folder_layout = QHBoxLayout()
        self.folder_btn = QPushButton("Choose Folder")
        self.folder_lbl = QLabel("No folder selected")
        self.folder_lbl.setStyleSheet("color: gray; font-style: italic;")
        folder_layout.addWidget(self.folder_btn)
        folder_layout.addWidget(self.folder_lbl)
        self.layout.addLayout(folder_layout)

        # 2. Load Data Button
        self.load_btn = QPushButton("Load Data")
        self.load_btn.setEnabled(False)
        self.layout.addWidget(self.load_btn)

        # 3. LIF Selection Row
        self.layout.addWidget(QLabel("Select LIF File:"))
        lif_layout = QHBoxLayout()
        self.prev_lif_btn = QPushButton("< Prev LIF")
        self.lif_combo = QComboBox()
        self.next_lif_btn = QPushButton("Next LIF >")
        lif_layout.addWidget(self.prev_lif_btn)
        lif_layout.addWidget(self.lif_combo)
        lif_layout.addWidget(self.next_lif_btn)
        self.layout.addLayout(lif_layout)

        # 4. Vein Selection Row
        self.layout.addWidget(QLabel("Select Vein:"))
        vein_layout = QHBoxLayout()
        self.prev_vein_btn = QPushButton("< Prev Vein")
        self.vein_combo = QComboBox()
        self.next_vein_btn = QPushButton("Next Vein >")
        vein_layout.addWidget(self.prev_vein_btn)
        vein_layout.addWidget(self.vein_combo)
        vein_layout.addWidget(self.next_vein_btn)
        self.layout.addLayout(vein_layout)

        # ==========================================
        # --- SEGMENTATION UI & PARAMETERS ---
        # ==========================================
        self.layout.addWidget(QLabel("--- Segmentation ---"))

        out_dir_layout = QHBoxLayout()
        out_dir_layout.addWidget(QLabel("Output Subfolder:"))
        self.out_dir_edit = QLineEdit("analysis")
        out_dir_layout.addWidget(self.out_dir_edit)
        self.layout.addLayout(out_dir_layout)

        self.load_masks_cb = QCheckBox("Auto-load existing segmentation masks")
        self.load_masks_cb.setChecked(False)
        self.layout.addWidget(self.load_masks_cb)

        # --- NEW: Wall Parameters Group ---
        self.wall_group = QGroupBox("Wall Parameters")
        wall_layout = QFormLayout()

        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.1, 5.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setValue(0.5)

        self.otsu_spin = QDoubleSpinBox()
        self.otsu_spin.setRange(0.1, 5.0)
        self.otsu_spin.setSingleStep(0.1)
        self.otsu_spin.setValue(0.5)

        wall_layout.addRow("Gamma:", self.gamma_spin)
        wall_layout.addRow("Otsu Multiplier:", self.otsu_spin)
        self.wall_group.setLayout(wall_layout)
        self.layout.addWidget(self.wall_group)

        # --- NEW: Chloroplast Parameters Group ---
        self.chlo_group = QGroupBox("Chloroplast Parameters (Omnipose)")
        chlo_layout = QFormLayout()

        self.gpu_cb = QCheckBox("Use GPU")
        self.gpu_cb.setChecked(False)  # Set to True if you want it on by default

        self.niter_spin = QSpinBox()
        self.niter_spin.setRange(0, 1000)
        self.niter_spin.setValue(20)

        chlo_layout.addRow("", self.gpu_cb)
        chlo_layout.addRow("niter (0 = False):", self.niter_spin)
        self.chlo_group.setLayout(chlo_layout)
        self.layout.addWidget(self.chlo_group)

        # --- Action Buttons ---
        self.test_vein_btn = QPushButton("Test Current Vein")
        self.seg_lif_btn = QPushButton("Segment Current LIF")
        self.seg_folder_btn = QPushButton("Segment Current Folder")

        self.layout.addWidget(self.test_vein_btn)
        self.layout.addWidget(self.seg_lif_btn)
        self.layout.addWidget(self.seg_folder_btn)

        self.layout.addStretch()

        # --- Connect Signals ---
        self.folder_btn.clicked.connect(self.select_folder)
        self.load_btn.clicked.connect(self.load_data)

        self.lif_combo.currentIndexChanged.connect(self.update_lif)
        self.prev_lif_btn.clicked.connect(lambda: self.step_combo(self.lif_combo, -1))
        self.next_lif_btn.clicked.connect(lambda: self.step_combo(self.lif_combo, 1))

        self.vein_combo.currentIndexChanged.connect(self.update_vein)
        self.prev_vein_btn.clicked.connect(lambda: self.step_combo(self.vein_combo, -1))
        self.next_vein_btn.clicked.connect(lambda: self.step_combo(self.vein_combo, 1))

        self.load_masks_cb.stateChanged.connect(self.update_vein)

        self.test_vein_btn.clicked.connect(self.test_current_vein)
        self.seg_lif_btn.clicked.connect(self.segment_current_lif)
        self.seg_folder_btn.clicked.connect(self.segment_current_folder)

    # --- Standard Interaction Logic ---
    def step_combo(self, combo: QComboBox, step: int):
        new_idx = combo.currentIndex() + step
        if 0 <= new_idx < combo.count():
            combo.setCurrentIndex(new_idx)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with LIF files")
        if folder:
            self.folder_lbl.setText(folder)
            max_len = 50
            if len(folder) > max_len:
                # Keep the first 20 chars, add "...", and keep the last 27 chars
                display_text = folder[:20] + "..." + folder[-27:]
            else:
                display_text = folder

            self.folder_lbl.setText(display_text)
            self.folder_lbl.setToolTip(
                folder
            )  # Allows user to hover and see the full path
            self.load_btn.setEnabled(True)

    def load_data(self):
        folder_path = Path(self.folder_lbl.text())
        if not folder_path.is_dir():
            return

        self.lif_files.clear()
        self.lif_combo.blockSignals(True)
        self.lif_combo.clear()

        for p in sorted(folder_path.glob("*.lif")):
            self.lif_files[p.name] = p
            self.lif_combo.addItem(p.name)

        self.lif_combo.blockSignals(False)

        if self.lif_combo.count() > 0:
            self.lif_combo.setCurrentIndex(0)
            self.update_lif()
        else:
            QMessageBox.warning(self, "No Files", "No .lif files found.")

    def update_lif(self):
        lif_name = self.lif_combo.currentText()
        if not lif_name:
            return

        self.current_lif = LifFile(self.lif_files[lif_name])

        self.vein_combo.blockSignals(True)
        self.vein_combo.clear()
        for img in self.current_lif.get_iter_image():
            self.vein_combo.addItem(img.name)

        self.vein_combo.blockSignals(False)

        if self.vein_combo.count() > 0:
            self.vein_combo.setCurrentIndex(0)
            self.update_vein()

    def update_vein(self):
        scene_idx = self.vein_combo.currentIndex()
        if scene_idx < 0 or not self.current_lif:
            return

        img = self.current_lif.get_image(scene_idx)
        z_dim, c_dim, y_dim, x_dim = img.dims.z, img.channels, img.dims.y, img.dims.x

        test_frame = np.array(img.get_frame(z=0, c=0))
        ch_data = [
            np.zeros((z_dim, y_dim, x_dim), dtype=test_frame.dtype) for _ in range(3)
        ]

        for z in range(z_dim):
            for c in range(min(c_dim, 3)):
                ch_data[c][z, :, :] = np.array(img.get_frame(z=z, c=c))

        self.current_wall_data = ch_data[0] if c_dim > 0 else None
        self.current_chlo_data = ch_data[1] if c_dim > 1 else None

        self.viewer.layers.clear()

        if c_dim > 2:
            self.viewer.add_image(
                ch_data[2], name="Brightfield", colormap="gray", visible=False
            )
        if c_dim > 1:
            self.viewer.add_image(
                ch_data[1], name="Chloroplast", colormap="green", blending="additive"
            )
        if c_dim > 0:
            self.viewer.add_image(
                ch_data[0],
                name="Cell Wall",
                colormap="red",
                blending="additive",
                opacity=0.6,
            )

        self.viewer.reset_view()

        # Load existing masks
        if self.load_masks_cb.isChecked():
            out_dir = self.get_output_dir()
            lif_name = self.lif_combo.currentText()
            vein_name = self.vein_combo.currentText()
            prefix = f"{lif_name}_{vein_name}"

            chlo_path = out_dir / "chlos" / f"{prefix}_chlo.tif"
            wall_path = out_dir / "walls" / f"{prefix}_wall.tif"

            if chlo_path.exists():
                try:
                    chlo_mask = tifffile.imread(chlo_path)
                    self.viewer.add_labels(
                        chlo_mask, name="Chlo Mask (Loaded)", opacity=0.5
                    )
                except Exception as e:
                    print(f"Failed to load {chlo_path}: {e}")

            if wall_path.exists():
                try:
                    wall_mask = tifffile.imread(wall_path)
                    self.viewer.add_image(
                        wall_mask,
                        name="Wall Mask (Loaded)",
                        colormap="yellow",
                        blending="additive",
                        opacity=1.0,
                        interpolation2d="linear",
                    )
                except Exception as e:
                    print(f"Failed to load {wall_path}: {e}")

    # --- SEGMENTATION LOGIC ---
    def get_output_dir(self):
        base_folder = Path(self.folder_lbl.text())
        subfolder_name = self.out_dir_edit.text().strip()
        if not subfolder_name:
            subfolder_name = "analysis"

        out_dir = base_folder / subfolder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    # --- NEW: Helper method to retrieve parameters from UI ---
    def get_seg_params(self):
        niter_val = self.niter_spin.value()

        # Evaluate Omnipose's GPU function if checked, otherwise pass False
        if self.gpu_cb.isChecked():
            gpu_param = use_gpu()
        else:
            gpu_param = False

        return {
            "gamma": self.gamma_spin.value(),
            "otsu_mult": self.otsu_spin.value(),
            "use_gpu": gpu_param,
            "niter": niter_val if niter_val > 0 else False,
        }

    def test_current_vein(self):
        if self.current_wall_data is None and self.current_chlo_data is None:
            QMessageBox.warning(self, "No Data", "No vein is currently loaded.")
            return

        lif_name = self.lif_combo.currentText()
        vein_name = self.vein_combo.currentText()
        prefix = f"{lif_name}_{vein_name}"
        params = self.get_seg_params()

        if self.current_chlo_data is not None:
            chlo_mask = seg_chlo(
                self.current_chlo_data,
                save_dir=None,
                filename_prefix=prefix,
                use_gpu=params["use_gpu"],
                niter=params["niter"],
            )
            self.viewer.add_labels(chlo_mask, name="Chlo Mask (Tested)", opacity=0.5)

        if self.current_wall_data is not None:
            wall_mask = seg_wall(
                self.current_wall_data,
                save_dir=None,
                filename_prefix=prefix,
                gamma=params["gamma"],
                otsu_mult=params["otsu_mult"],
            )
            self.viewer.add_image(
                wall_mask,
                name="Wall Mask (Tested)",
                colormap="yellow",
                blending="additive",
                opacity=1.0,
                interpolation2d="linear",
            )

    def _process_single_lif(self, lif_path, out_dir):
        lif_obj = LifFile(lif_path)
        params = self.get_seg_params()

        for scene_idx, img in enumerate(lif_obj.get_iter_image()):
            z_dim, c_dim, y_dim, x_dim = (
                img.dims.z,
                img.channels,
                img.dims.y,
                img.dims.x,
            )
            prefix = f"{lif_path.name}_{img.name}"

            if c_dim > 0:
                wall_data = np.zeros((z_dim, y_dim, x_dim), dtype=np.uint16)
                for z in range(z_dim):
                    wall_data[z, :, :] = np.array(img.get_frame(z=z, c=0))
                seg_wall(
                    wall_data,
                    save_dir=out_dir,
                    filename_prefix=prefix,
                    gamma=params["gamma"],
                    otsu_mult=params["otsu_mult"],
                )

            if c_dim > 1:
                chlo_data = np.zeros((z_dim, y_dim, x_dim), dtype=np.uint16)
                for z in range(z_dim):
                    chlo_data[z, :, :] = np.array(img.get_frame(z=z, c=1))
                seg_chlo(
                    chlo_data,
                    save_dir=out_dir,
                    filename_prefix=prefix,
                    use_gpu=params["use_gpu"],
                    niter=params["niter"],
                )

    def segment_current_lif(self):
        lif_name = self.lif_combo.currentText()
        if not lif_name:
            return

        out_dir = self.get_output_dir()
        lif_path = self.lif_files[lif_name]

        self.seg_lif_btn.setText("Processing...")
        self.seg_lif_btn.setEnabled(False)

        self._process_single_lif(lif_path, out_dir)

        self.seg_lif_btn.setEnabled(True)
        self.seg_lif_btn.setText("Segment Current LIF")
        QMessageBox.information(self, "Done", f"Finished segmenting {lif_name}")

    def segment_current_folder(self):
        if not self.lif_files:
            return

        out_dir = self.get_output_dir()

        self.seg_folder_btn.setText("Processing...")
        self.seg_folder_btn.setEnabled(False)

        for lif_name, lif_path in self.lif_files.items():
            self._process_single_lif(lif_path, out_dir)

        self.seg_folder_btn.setEnabled(True)
        self.seg_folder_btn.setText("Segment Current Folder")
        QMessageBox.information(self, "Done", "Finished segmenting entire folder.")
