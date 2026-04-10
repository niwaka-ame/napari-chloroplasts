import napari
import numpy as np
import tifffile
import networkx as nx
import csv
from pathlib import Path
from readlif.reader import LifFile
from skimage.measure import regionprops, label as sk_label
from skimage.segmentation import find_boundaries
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QLabel,
    QFileDialog,
    QMessageBox,
    QRadioButton,
    QButtonGroup,
    QCheckBox,
    QApplication,
)

# --- USER PROVIDED LOGIC ---


def calculate_iom(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    min_area = min(mask_a.sum(), mask_b.sum())
    if min_area == 0:
        return 0.0
    return intersection / min_area


def extract_all_chloroplasts_undirected(cell_mask, chloro_stack, iom_threshold=0.6):
    num_z = len(chloro_stack)
    valid_nodes = {}

    for z in range(num_z):
        valid_nodes[z] = []
        for region in regionprops(chloro_stack[z]):
            c_mask = chloro_stack[z] == region.label
            if calculate_iom(cell_mask, c_mask) >= iom_threshold:
                valid_nodes[z].append(
                    {"label": region.label, "area": region.area, "mask": c_mask}
                )

    G = nx.Graph()
    for z, chloros in valid_nodes.items():
        for c in chloros:
            G.add_node((z, c["label"]), area=c["area"], mask=c["mask"])

    for z in range(num_z - 1):
        for c1 in valid_nodes[z]:
            for c2 in valid_nodes[z + 1]:
                if calculate_iom(c1["mask"], c2["mask"]) >= iom_threshold:
                    G.add_edge((z, c1["label"]), (z + 1, c2["label"]))

    reliable, unreliable = [], []

    for cc in nx.connected_components(G):
        if len(cc) < 2:
            # unreliable.append(list(cc))
            continue

        z_indices = [node[0] for node in cc]
        if len(z_indices) != len(set(z_indices)):
            unreliable.append(list(cc))
            continue

        sub_graph = G.subgraph(cc)
        peak_node = max(cc, key=lambda node: sub_graph.nodes[node]["area"])
        reliable.append(
            {
                "peak_z": peak_node[0],
                "peak_mask": sub_graph.nodes[peak_node]["mask"],
            }
        )

    return reliable, unreliable


# --- NAPARI PLUGIN UI ---


class LineageCorrectorWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.lif_files = {}
        self.base_folder = None
        self.current_lif = None

        # State Data
        self.full_chlo_raw = None
        self.full_chlo_mask = None
        self.full_cell_mask = None
        self.available_cells = []

        # --- NEW: Unsaved Changes & Navigation Tracking ---
        self.unsaved_changes = False
        self.last_lif_idx = -1
        self.last_vein_idx = -1
        self.last_cell_idx = -1

        # Active Cell Cropped Data
        self.current_crop_chlo_mask = None
        self.current_crop_bounds = None
        self.target_cell_bool = None
        self.orig_active_mask_bool = None

        # Graph & Editing State
        self.reliable_chlos = []
        self.unreliable_chlos = []
        self.current_unreliable_idx = 0
        self.undo_stack = []
        self.merge_source_id = None

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Create a flag to track if the user actually used the brush
        self.mask_modified = False

        # 1. Folder Selection
        folder_layout = QHBoxLayout()
        self.folder_btn = QPushButton("Choose Folder")
        self.folder_lbl = QLabel("No folder selected")
        self.folder_lbl.setStyleSheet("color: gray; font-style: italic;")
        folder_layout.addWidget(self.folder_btn)
        folder_layout.addWidget(self.folder_lbl)
        self.layout.addLayout(folder_layout)

        self.load_btn = QPushButton("Load Data")
        self.load_btn.setEnabled(False)
        self.layout.addWidget(self.load_btn)

        # 2. LIF & Vein Selection
        self.layout.addWidget(QLabel("Select LIF File:"))
        lif_layout = QHBoxLayout()
        self.btn_prev_lif = QPushButton("< Prev")
        self.lif_combo = QComboBox()
        self.btn_next_lif = QPushButton("Next >")
        lif_layout.addWidget(self.btn_prev_lif)
        lif_layout.addWidget(self.lif_combo)
        lif_layout.addWidget(self.btn_next_lif)
        self.layout.addLayout(lif_layout)

        self.layout.addWidget(QLabel("Select Vein:"))
        vein_layout = QHBoxLayout()
        self.btn_prev_vein = QPushButton("< Prev")
        self.vein_combo = QComboBox()
        self.btn_next_vein = QPushButton("Next >")
        vein_layout.addWidget(self.btn_prev_vein)
        vein_layout.addWidget(self.vein_combo)
        vein_layout.addWidget(self.btn_next_vein)
        self.layout.addLayout(vein_layout)

        # 3. Cell Selection
        self.layout.addWidget(QLabel("--- 1. Select Cell ---"))
        cell_layout = QHBoxLayout()
        self.btn_prev_cell = QPushButton("< Prev")
        self.cell_combo = QComboBox()
        self.btn_next_cell = QPushButton("Next >")
        cell_layout.addWidget(self.btn_prev_cell)
        cell_layout.addWidget(self.cell_combo)
        cell_layout.addWidget(self.btn_next_cell)
        self.layout.addLayout(cell_layout)

        # 4. Explicit Editing Modes
        self.layout.addWidget(QLabel("--- 2. Edit Modes ---"))

        mode_layout = QHBoxLayout()
        self.rad_normal = QRadioButton("Normal")
        self.rad_merge = QRadioButton("Merge")
        self.rad_split = QRadioButton("Split")
        self.rad_add = QRadioButton("Add")
        self.rad_delete = QRadioButton("Delete")  # NEW: Delete button
        self.rad_normal.setChecked(True)

        self.mode_group = QButtonGroup()
        for rad in [
            self.rad_normal,
            self.rad_merge,
            self.rad_split,
            self.rad_add,
            self.rad_delete,
        ]:
            self.mode_group.addButton(rad)
            mode_layout.addWidget(rad)
            rad.toggled.connect(self.on_mode_changed)
        self.layout.addLayout(mode_layout)

        self.btn_undo = QPushButton("↶ Undo Last Edit")
        self.layout.addWidget(self.btn_undo)

        self.btn_revert = QPushButton("⏪ Revert All Cell Edits")
        self.btn_revert.setStyleSheet("background-color: #d9534f; color: white;")
        self.layout.addWidget(self.btn_revert)

        # Visibility Filters for Graph
        self.layout.addWidget(QLabel("--- 3. Unreliable Components ---"))
        self.rad_all = QRadioButton("Show All Masks")
        self.rad_unreliable = QRadioButton("Show Current Unreliable Only")
        self.rad_all.setChecked(True)

        self.vis_group = QButtonGroup()
        self.vis_group.addButton(self.rad_all)
        self.vis_group.addButton(self.rad_unreliable)
        self.layout.addWidget(self.rad_all)
        self.layout.addWidget(self.rad_unreliable)

        # Navigation
        nav_layout = QHBoxLayout()
        self.btn_prev_unrel = QPushButton("< Prev")
        self.lbl_unrel_status = QLabel("0 of 0")
        self.btn_next_unrel = QPushButton("Next >")
        nav_layout.addWidget(self.btn_prev_unrel)
        nav_layout.addWidget(self.lbl_unrel_status)
        nav_layout.addWidget(self.btn_next_unrel)
        self.layout.addLayout(nav_layout)

        # Actions
        self.btn_update_graph = QPushButton("↻ Update Graph (Apply Edits)")
        self.btn_update_graph.setStyleSheet("background-color: #4CAF50; color: white;")
        self.layout.addWidget(self.btn_update_graph)

        self.global_status_lbl = QLabel("Ready.")
        self.layout.addWidget(self.global_status_lbl)

        self.layout.addWidget(QLabel("--- 4. Save ---"))
        self.btn_save_corrected = QPushButton("💾 Save Corrected Chlos (3D)")
        self.layout.addWidget(self.btn_save_corrected)

        # 5. Export
        self.layout.addWidget(QLabel("--- 5. Export ---"))

        # First row of checkboxes
        export_opt_layout1 = QHBoxLayout()
        self.chk_resolved_only = QCheckBox("Resolved cells only")
        self.chk_resolved_only.setChecked(True)
        self.chk_microns = QCheckBox("Export in microns")
        self.chk_microns.setChecked(True)
        export_opt_layout1.addWidget(self.chk_resolved_only)
        export_opt_layout1.addWidget(self.chk_microns)
        self.layout.addLayout(export_opt_layout1)

        # Second row of checkboxes
        export_opt_layout2 = QHBoxLayout()
        self.chk_export_chlo_rows = QCheckBox("Export each chloroplast as one row")
        self.chk_export_chlo_rows.setChecked(True)
        export_opt_layout2.addWidget(self.chk_export_chlo_rows)
        self.layout.addLayout(export_opt_layout2)

        # Export buttons
        export_btn_layout = QHBoxLayout()
        self.combo_export_scope = QComboBox()
        self.combo_export_scope.addItems(
            ["Current Vein", "Current LIF", "Entire Folder"]
        )
        self.btn_export_csv = QPushButton("Export to CSV")
        export_btn_layout.addWidget(self.combo_export_scope)
        export_btn_layout.addWidget(self.btn_export_csv)
        self.layout.addLayout(export_btn_layout)

        self.layout.addStretch()

        # Connect Signals
        self.folder_btn.clicked.connect(self.select_folder)
        self.load_btn.clicked.connect(self.load_data)

        # Prev/Next connections
        self.btn_prev_lif.clicked.connect(lambda: self.step_combo(self.lif_combo, -1))
        self.btn_next_lif.clicked.connect(lambda: self.step_combo(self.lif_combo, 1))
        self.btn_prev_vein.clicked.connect(lambda: self.step_combo(self.vein_combo, -1))
        self.btn_next_vein.clicked.connect(lambda: self.step_combo(self.vein_combo, 1))
        self.btn_prev_cell.clicked.connect(lambda: self.step_combo(self.cell_combo, -1))
        self.btn_next_cell.clicked.connect(lambda: self.step_combo(self.cell_combo, 1))

        # UPDATED: Safety wrapper signals
        self.lif_combo.currentIndexChanged.connect(self.handle_lif_change)
        self.vein_combo.currentIndexChanged.connect(self.handle_vein_change)
        self.cell_combo.currentIndexChanged.connect(self.handle_cell_change)

        self.btn_undo.clicked.connect(self.undo_edit)
        self.btn_revert.clicked.connect(self.revert_to_original)
        self.rad_all.clicked.connect(self.on_visibility_toggle)
        self.rad_unreliable.clicked.connect(self.on_visibility_toggle)
        self.btn_prev_unrel.clicked.connect(lambda: self.step_unreliable(-1))
        self.btn_next_unrel.clicked.connect(lambda: self.step_unreliable(1))

        self.btn_update_graph.clicked.connect(self.update_graph_action)
        self.btn_save_corrected.clicked.connect(self.save_corrected)
        self.btn_export_csv.clicked.connect(self.export_data)

    # --- Navigation Safety Wrappers ---
    def prompt_unsaved(self):
        """Checks for unsaved changes and prompts the user. Returns True if safe to proceed."""
        # Catch any pending brush strokes
        if getattr(self, "mask_modified", False):
            self.unsaved_changes = True

        if getattr(self, "unsaved_changes", False):
            reply = QMessageBox.question(
                self,
                "Unsaved Edits",
                "You have unsaved changes in the current cell. Navigating away will discard them.\n\nAre you sure you want to continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            return reply == QMessageBox.Yes
        return True

    def handle_lif_change(self, index):
        if index < 0 or index == self.last_lif_idx:
            return
        if not self.prompt_unsaved():
            self.lif_combo.blockSignals(True)
            self.lif_combo.setCurrentIndex(self.last_lif_idx)
            self.lif_combo.blockSignals(False)
            return
        self.last_lif_idx = index
        self.unsaved_changes = False
        self.update_lif()

    def handle_vein_change(self, index):
        if index < 0 or index == self.last_vein_idx:
            return
        if not self.prompt_unsaved():
            self.vein_combo.blockSignals(True)
            self.vein_combo.setCurrentIndex(self.last_vein_idx)
            self.vein_combo.blockSignals(False)
            return
        self.last_vein_idx = index
        self.unsaved_changes = False
        self.load_vein_data()

    def handle_cell_change(self, index):
        if index < 0 or index == self.last_cell_idx:
            return
        if not self.prompt_unsaved():
            self.cell_combo.blockSignals(True)
            self.cell_combo.setCurrentIndex(self.last_cell_idx)
            self.cell_combo.blockSignals(False)
            return
        self.last_cell_idx = index
        self.unsaved_changes = False
        self.process_selected_cell()

    # --- Combobox Helper ---
    def step_combo(self, combo: QComboBox, step: int):
        new_idx = combo.currentIndex() + step
        if 0 <= new_idx < combo.count():
            combo.setCurrentIndex(new_idx)

    # --- Loading & Initialization ---
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with LIF files")
        if folder:
            # 1. Save the REAL, full path to the internal variable
            self.base_folder = Path(folder)

            # 2. Create a FAKE, truncated path just for the UI
            max_len = 30
            if len(folder) > max_len:
                display_text = folder[:10] + "..." + folder[-17:]
            else:
                display_text = folder

            # 3. Apply the fake text to the label, but keep the real path in the tooltip
            self.folder_lbl.setText(display_text)
            self.folder_lbl.setToolTip(folder)

            self.load_btn.setEnabled(True)

    def load_data(self):
        # Added .is_dir() to ensure stability based on previous edits
        if not self.base_folder or not self.base_folder.is_dir():
            return
        self.lif_files.clear()
        self.lif_combo.blockSignals(True)
        self.lif_combo.clear()

        for p in sorted(self.base_folder.glob("*.lif")):
            self.lif_files[p.name] = p

        self.lif_combo.blockSignals(False)
        self.lif_combo.addItems(list(self.lif_files.keys()))

        if self.lif_combo.count() > 0:
            self.last_lif_idx = self.lif_combo.currentIndex()
            self.update_lif()

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
            self.last_vein_idx = self.vein_combo.currentIndex()
            self.load_vein_data()

    def load_vein_data(self):
        scene_idx = self.vein_combo.currentIndex()
        if scene_idx < 0 or not self.current_lif:
            return

        img = self.current_lif.get_image(scene_idx)
        z_dim, c_dim, y_dim, x_dim = img.dims.z, img.channels, img.dims.y, img.dims.x
        chlo_idx = 1 if c_dim > 1 else 0

        self.full_chlo_raw = np.zeros((z_dim, y_dim, x_dim), dtype=np.uint16)
        for z in range(z_dim):
            self.full_chlo_raw[z, :, :] = np.array(img.get_frame(z=z, c=chlo_idx))

        prefix = f"{self.lif_combo.currentText()}_{self.vein_combo.currentText()}"

        cell_path = self.base_folder / "analysis" / "cells" / f"{prefix}_cells.tif"
        if cell_path.exists():
            self.full_cell_mask = tifffile.imread(cell_path)
            self.available_cells = np.unique(self.full_cell_mask)
            self.available_cells = self.available_cells[self.available_cells > 0]
        else:
            self.full_cell_mask = None
            QMessageBox.warning(self, "Missing Data", "Cell mask not found!")

        corrected_path = (
            self.base_folder / "analysis" / "chlos_corrected" / f"{prefix}_chlo.tif"
        )
        raw_path = self.base_folder / "analysis" / "chlos" / f"{prefix}_chlo.tif"

        if corrected_path.exists():
            self.full_chlo_mask = tifffile.imread(corrected_path)
            print("Loaded CORRECTED Chlos")
        elif raw_path.exists():
            self.full_chlo_mask = tifffile.imread(raw_path)
            print("Loaded RAW Chlos")
        else:
            self.full_chlo_mask = None
            QMessageBox.warning(
                self, "Missing Data", f"Chloroplast masks not found for {prefix}!"
            )

        self.cell_combo.blockSignals(True)
        self.cell_combo.clear()
        for c_id in self.available_cells:
            self.cell_combo.addItem(f"Cell {c_id}")
        self.cell_combo.blockSignals(False)

        if self.cell_combo.count() > 0:
            self.last_cell_idx = self.cell_combo.currentIndex()
            self.process_selected_cell()

    # --- Processing & Graph Extraction ---
    def process_selected_cell(self):
        if (
            self.cell_combo.currentIndex() < 0
            or self.full_cell_mask is None
            or self.full_chlo_mask is None
        ):
            return

        cell_id = int(self.cell_combo.currentText().split(" ")[1])
        props = regionprops((self.full_cell_mask == cell_id).astype(np.uint8))
        if not props:
            return

        min_row, min_col, max_row, max_col = props[0].bbox
        pad = 20
        rmin, rmax = max(0, min_row - pad), min(
            self.full_cell_mask.shape[0], max_row + pad
        )
        cmin, cmax = max(0, min_col - pad), min(
            self.full_cell_mask.shape[1], max_col + pad
        )

        self.current_crop_bounds = (rmin, rmax, cmin, cmax)
        crop_cell_mask = self.full_cell_mask[rmin:rmax, cmin:cmax]
        self.current_crop_chlo_mask = self.full_chlo_mask[
            :, rmin:rmax, cmin:cmax
        ].copy()
        self.target_cell_bool = crop_cell_mask == cell_id

        self.undo_stack.clear()

        # Destroy the old layer so it can't be saved to the new cell
        if "Editable Chlo Masks" in self.viewer.layers:
            self.viewer.layers.remove("Editable Chlo Masks")

        self.run_graph_extraction()

        # Block signals so checking the button doesn't trigger on_visibility_toggle
        self.rad_all.blockSignals(True)
        self.rad_all.setChecked(True)
        self.rad_all.blockSignals(False)

        self.rad_normal.setChecked(True)
        self.render_viewer()

    def run_graph_extraction(self):
        rel, unrel = extract_all_chloroplasts_undirected(
            self.target_cell_bool, self.current_crop_chlo_mask, iom_threshold=0.6
        )
        self.reliable_chlos = rel
        self.unreliable_chlos = unrel
        self.current_unreliable_idx = 0
        self.update_ui_state()

    def update_ui_state(self):
        if len(self.unreliable_chlos) == 0:
            self.rad_all.setChecked(True)
            self.rad_unreliable.setEnabled(False)
            self.btn_prev_unrel.setEnabled(False)
            self.btn_next_unrel.setEnabled(False)
            self.lbl_unrel_status.setText("All resolved!")
            self.global_status_lbl.setText("All resolved! Save whenever ready.")
        else:
            self.rad_unreliable.setEnabled(True)
            self.btn_prev_unrel.setEnabled(True)
            self.btn_next_unrel.setEnabled(True)
            self.lbl_unrel_status.setText(
                f"Component {self.current_unreliable_idx + 1} / {len(self.unreliable_chlos)}"
            )

    # --- Editing Interactions ---

    def save_history(self, layer):
        """Saves a snapshot of the mask array and its context for the Undo button."""
        self.undo_stack.append(
            {
                "layer_data": layer.data.copy(),
                "master_mask": self.current_crop_chlo_mask.copy(),
                "orig_bool": (
                    self.orig_active_mask_bool.copy()
                    if self.orig_active_mask_bool is not None
                    else None
                ),
                "is_all_mode": self.rad_all.isChecked(),
            }
        )
        if len(self.undo_stack) > 15:  # Keep memory clean
            self.undo_stack.pop(0)

    def undo_edit(self):
        if not self.undo_stack:
            self.global_status_lbl.setText("Nothing to undo.")
            return

        if "Editable Chlo Masks" in self.viewer.layers:
            layer = self.viewer.layers["Editable Chlo Masks"]
            state = self.undo_stack.pop()

            # 1. Restore the background master mask to prevent wiping hidden masks
            self.current_crop_chlo_mask = state["master_mask"]

            # 2. Restore the correct boolean footprint mapping
            self.orig_active_mask_bool = state["orig_bool"]

            # 3. Sync the UI visibility radio buttons
            # (We block signals so checking a button doesn't prematurely trigger an apply_edits_to_master)
            self.rad_all.blockSignals(True)
            self.rad_unreliable.blockSignals(True)
            if state["is_all_mode"]:
                self.rad_all.setChecked(True)
            else:
                self.rad_unreliable.setChecked(True)
            self.rad_all.blockSignals(False)
            self.rad_unreliable.blockSignals(False)

            # 4. Restore the visual napari layer array
            layer.data = state["layer_data"].copy()
            self.merge_source_id = None

            self.global_status_lbl.setText(
                "Undo successful. (Press 'Update Graph' to refresh peaks)"
            )

    def revert_to_original(self):
        if self.cell_combo.currentIndex() < 0:
            return

        # Add a quick safety check so they don't accidentally wipe 20 minutes of work
        reply = QMessageBox.question(
            self,
            "Revert Edits",
            "Are you sure you want to discard all unsaved edits for this cell?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.unsaved_changes = False
            # Re-running this perfectly overwrites the current cropped workspace
            # with the original, unedited master mask for this cell.
            self.process_selected_cell()
            self.global_status_lbl.setText("Reverted cell to original masks.")

    def on_mode_changed(self):
        if "Editable Chlo Masks" not in self.viewer.layers:
            return
        layer = self.viewer.layers["Editable Chlo Masks"]
        self.merge_source_id = None  # Reset partial merges

        if self.rad_normal.isChecked():
            layer.mode = "pan_zoom"
            self.global_status_lbl.setText("Normal Mode: View and pan safely.")
        elif self.rad_merge.isChecked():
            layer.mode = "pan_zoom"  # Disable painting so click triggers custom hook
            self.global_status_lbl.setText(
                "Merge Mode: Click 1st mask, then click 2nd to merge."
            )
        elif self.rad_split.isChecked():
            layer.mode = "erase"
            layer.brush_size = 1
            self.global_status_lbl.setText(
                "Split Mode: Draw to erase and cut masks in half."
            )
        elif self.rad_add.isChecked():
            layer.mode = "paint"
            layer.brush_size = 1
            layer.selected_label = max(self.full_chlo_mask.max(), layer.data.max()) + 1
            self.global_status_lbl.setText(
                f"Add Mode: Auto-assigned new ID {layer.selected_label}."
            )
        elif self.rad_delete.isChecked():
            layer.mode = "pan_zoom"
            self.global_status_lbl.setText(
                "Delete Mode: Click a mask to delete it on the current Z-slice."
            )

    def custom_mouse_hook(self, layer, event):
        """Handles manual history saving, custom consecutive merging, auto-splitting, and deletion."""

        # --- NEW: Custom DELETE Logic ---
        if self.rad_delete.isChecked() and event.button == 1:
            coords = tuple(
                int(np.round(c)) for c in layer.world_to_data(event.position)
            )

            # Check bounds
            if all(0 <= c < m for c, m in zip(coords, layer.data.shape)):
                clicked_id = layer.data[coords]
                if clicked_id > 0:
                    self.save_history(layer)

                    # Overwrite target mask with 0 (background) ONLY on the active Z-stack
                    new_data = layer.data.copy()
                    z_idx = coords[0]  # The Z-slice where the user clicked

                    # Restrict the reassignment to only the current Z-slice
                    slice_data = new_data[z_idx]
                    slice_data[slice_data == clicked_id] = 0

                    layer.data = new_data
                    self.mask_modified = True
                    self.global_status_lbl.setText(
                        f"Deleted ID {clicked_id} on Z-slice {z_idx}."
                    )
            yield
            return
        # --------------------------------

        # Custom MERGE Logic
        if self.rad_merge.isChecked() and event.button == 1:
            coords = tuple(
                int(np.round(c)) for c in layer.world_to_data(event.position)
            )

            # Check bounds
            if all(0 <= c < m for c, m in zip(coords, layer.data.shape)):
                clicked_id = layer.data[coords]
                if clicked_id > 0:
                    if self.merge_source_id is None:
                        self.merge_source_id = clicked_id
                        self.global_status_lbl.setText(
                            f"Merge: Picked ID {clicked_id}. Click target to merge."
                        )
                    else:
                        if self.merge_source_id != clicked_id:
                            self.save_history(layer)

                            # Overwrite target mask with source ID ONLY on the active Z-stack
                            new_data = layer.data.copy()
                            z_idx = coords[0]  # The Z-slice where the user clicked

                            # Restrict the reassignment to only the current Z-slice
                            slice_data = new_data[z_idx]
                            slice_data[slice_data == clicked_id] = self.merge_source_id

                            layer.data = new_data
                            self.global_status_lbl.setText(
                                f"Merged ID {clicked_id} into {self.merge_source_id} on Z-slice {z_idx}."
                            )
                        self.merge_source_id = None  # Reset
                        self.mask_modified = True
            yield
            return

        is_erase = layer.mode == "erase"
        is_paint = layer.mode == "paint"

        # Save history if drawing/erasing starts
        if is_paint or is_erase:
            self.save_history(layer)

        yield  # Wait for mouse drag to finish

        # Consume drag events to let Napari draw without re-triggering history
        while event.type == "mouse_move":
            yield

        # Post-Erase Auto-Split Logic
        if is_erase and self.undo_stack:
            last_state = self.undo_stack[-1]
            prev_data = last_state["layer_data"]  # Extract the actual NumPy array!
            curr_data = layer.data

            # Find the pixels that were erased
            changed = prev_data != curr_data

            # Identify exactly which Z-slices were affected by the erase stroke
            changed_z_indices = np.unique(np.where(changed)[0])

            new_data = curr_data.copy()
            split_occurred = False

            # Process ONLY the Z-slices where the erase happened
            for z_idx in changed_z_indices:
                # Find which IDs were touched on this specific slice
                touched_ids = np.unique(prev_data[z_idx][changed[z_idx]])

                for tid in touched_ids:
                    if tid == 0:
                        continue

                    # Isolate the mask to ONLY this Z-slice
                    bin_mask = curr_data[z_idx] == tid
                    if not np.any(bin_mask):
                        continue

                    # Check for disconnected components in 2D (connectivity=1 for 4-way pixels)
                    labeled_components, num_features = sk_label(
                        bin_mask, return_num=True, connectivity=1
                    )

                    if num_features > 1:
                        split_occurred = True
                        props = regionprops(labeled_components)
                        # Sort components by area so the largest piece is evaluated first
                        props.sort(key=lambda x: x.area, reverse=True)

                        min_area = 10  # Configurable debris threshold

                        for i, p in enumerate(props):
                            if p.area < min_area:
                                # DEBRIS: Erase it completely by setting to background (0)
                                new_data[z_idx][labeled_components == p.label] = 0
                            elif i > 0:
                                # VALID SEVERED PIECE: Assign a new unique ID
                                new_id = (
                                    max(self.full_chlo_mask.max(), new_data.max()) + 1
                                )
                                new_data[z_idx][labeled_components == p.label] = new_id

            if split_occurred:
                layer.data = new_data
                self.global_status_lbl.setText(
                    "Split Mode: Erase completed and severed masks automatically assigned new IDs."
                )

        # Auto-increment label for consecutive Add strokes
        if is_paint and self.rad_add.isChecked():
            layer.selected_label = max(self.full_chlo_mask.max(), layer.data.max()) + 1
            self.global_status_lbl.setText(
                f"Add Mode: Stroke finished. Auto-assigned next new ID {layer.selected_label}."
            )

    # --- Visbility & Saving ---
    def apply_edits_to_master(self):
        # 1. Immediate Bypass: If Napari didn't register a brush stroke, do nothing.
        if getattr(self, "mask_modified", False) is False:
            return

        self.unsaved_changes = True

        if "Editable Chlo Masks" not in self.viewer.layers:
            return

        edited_data = self.viewer.layers["Editable Chlo Masks"].data

        if self.orig_active_mask_bool is not None:
            # --- COMPONENT ONLY MODE ---
            # Erase the original footprint of this component
            self.current_crop_chlo_mask[self.orig_active_mask_bool] = 0

            mask_new = edited_data > 0

            # Prevent overwriting hidden reliable components
            # Find where the user painted outside the original footprint
            expansion_mask = mask_new & ~self.orig_active_mask_bool

            # Check if that expansion hits an existing (hidden) component in the master crop
            collisions = self.current_crop_chlo_mask[expansion_mask] > 0

            if np.any(collisions):
                print(
                    "Warning: Edits overlap with hidden components. Overwrite prevented in those voxels."
                )

                # Create a safe expansion mask where collisions are mapped out
                safe_expansion = expansion_mask.copy()
                safe_expansion[expansion_mask] = ~collisions

                # Rebuild the final mask: original footprint + safe expansions only
                safe_mask_new = (mask_new & self.orig_active_mask_bool) | safe_expansion

                # Paste using the corrected boolean mask
                self.current_crop_chlo_mask[safe_mask_new] = edited_data[safe_mask_new]
                self.global_status_lbl.setText(
                    "Applied edits (prevented collision with hidden masks)."
                )
            else:
                # Paste the exact IDs directly if there are no collisions
                self.current_crop_chlo_mask[mask_new] = edited_data[mask_new]
                self.global_status_lbl.setText("Applied edits safely.")

        else:
            # --- ALL MASKS MODE ---
            if self.current_crop_chlo_mask.shape == edited_data.shape:
                self.current_crop_chlo_mask = edited_data.copy()

        # 2. Reset the flag
        self.mask_modified = False

    def on_visibility_toggle(self):
        self.apply_edits_to_master()
        self.render_viewer()

    def step_unreliable(self, step):
        if not self.unreliable_chlos:
            return
        self.apply_edits_to_master()
        self.current_unreliable_idx = (self.current_unreliable_idx + step) % len(
            self.unreliable_chlos
        )
        self.rad_unreliable.setChecked(True)
        self.update_ui_state()
        self.render_viewer()

    def update_graph_action(self):
        self.apply_edits_to_master()
        self.run_graph_extraction()
        self.render_viewer()

    def render_viewer(self):
        self.viewer.layers.clear()
        rmin, rmax, cmin, cmax = self.current_crop_bounds
        crop_chlo_raw = self.full_chlo_raw[:, rmin:rmax, cmin:cmax]

        self.viewer.add_image(
            crop_chlo_raw,
            name="Raw Chlo (Cropped)",
            colormap="red",
            blending="additive",
        )

        # --- NEW: Containers for Unreliable Z-specific text ---
        unrel_text_coords = []
        unrel_text_labels = []

        if self.rad_all.isChecked() or len(self.unreliable_chlos) == 0:
            display_data = self.current_crop_chlo_mask.copy()
            self.orig_active_mask_bool = None
        else:
            display_data = np.zeros_like(self.current_crop_chlo_mask)
            nodes = self.unreliable_chlos[self.current_unreliable_idx]
            for z, label_id in nodes:
                mask = self.current_crop_chlo_mask[z] == label_id
                display_data[z][mask] = label_id

                # Calculate centroid to place the ID text on this specific Z-slice
                props = regionprops(mask.astype(np.uint8))
                if props:
                    y, x = props[0].centroid
                    unrel_text_coords.append([z, y, x])
                    unrel_text_labels.append(str(label_id))  # Just the ID number

            self.orig_active_mask_bool = display_data > 0

        edit_layer = self.viewer.add_labels(
            display_data, name="Editable Chlo Masks", opacity=0.7
        )

        # Explicitly lock the brush footprint strictly to the 2D viewing plane (safety net)
        edit_layer.n_edit_dimensions = 2

        # Attach the custom mouse hook to our new layer
        edit_layer.mouse_drag_callbacks.append(self.custom_mouse_hook)

        # Attach the data listener directly to the newly created layer
        def on_data_change(event):
            self.mask_modified = True
            self.unsaved_changes = True

        edit_layer.events.data.connect(on_data_change)
        edit_layer.events.paint.connect(on_data_change)

        cell_bound_2d = find_boundaries(self.target_cell_bool, mode="outer")
        cell_contour_3d = np.zeros_like(crop_chlo_raw, dtype=np.uint8)
        cell_contour_3d[:] = cell_bound_2d

        cell_layer = self.viewer.add_labels(
            cell_contour_3d, name="Cell Contour", opacity=1.0
        )
        try:
            cell_layer.color_mode = "direct"
            cell_layer.color = {1: "white"}
        except AttributeError:
            pass

        peak_contours_3d = np.zeros_like(self.current_crop_chlo_mask, dtype=np.uint16)
        for i, chlo in enumerate(self.reliable_chlos, start=1):
            bound = find_boundaries(chlo["peak_mask"], mode="outer")
            peak_contours_3d[:, bound] = i
        self.viewer.add_labels(peak_contours_3d, name="Reliable Peaks", opacity=1.0)

        # --- NEW: Reliable Info Text Layer ---
        text_coords = []
        text_labels = []

        # Get the total number of Z-slices in the current volume
        z_dim = self.current_crop_chlo_mask.shape[0]

        for i, chlo in enumerate(self.reliable_chlos, start=1):
            mask_2d = chlo["peak_mask"]
            area = mask_2d.sum()

            # Find the centroid of the peak 2D mask to place the text
            props = regionprops(mask_2d.astype(np.uint8))
            if props:
                y, x = props[0].centroid
                label_text = f"ID:{i} | A:{area}"

                # Duplicate this text point across EVERY Z-slice
                for z in range(z_dim):
                    text_coords.append([z, y, x])
                    text_labels.append(label_text)

        if text_coords:
            text_kwargs = {
                "string": "{label}",
                "color": "yellow",
                "size": 10,
                "anchor": "center",
            }
            properties = {"label": text_labels}

            # Add an invisible points layer purely to host the 3D text
            self.viewer.add_points(
                np.array(text_coords),
                properties=properties,
                text=text_kwargs,
                size=0,  # Hides the points themselves
                name="Reliable Info",
                visible=False,  # Default hidden
            )
        # -------------------------------------

        # --- NEW: Unreliable IDs Text Layer (Z-specific) ---
        if unrel_text_coords:
            unrel_text_kwargs = {
                "string": "{label}",
                "color": "white",  # White usually contrasts well against the raw red background
                "size": 12,
                "anchor": "center",
            }
            unrel_properties = {"label": unrel_text_labels}

            self.viewer.add_points(
                np.array(unrel_text_coords),
                properties=unrel_properties,
                text=unrel_text_kwargs,
                size=0,  # Hides the points
                name="Unreliable IDs",
                visible=True,  # Default visible!
            )
        # ---------------------------------------------------

        self.on_mode_changed()  # Re-apply the selected editing mode to the fresh layer

        # FORCE FOCUS: Make the editable layer immediately active so user doesn't have to click it
        self.viewer.layers.selection.active = edit_layer

        self.viewer.reset_view()

    def save_corrected(self):
        if self.current_crop_chlo_mask is None:
            return
        self.apply_edits_to_master()

        rmin, rmax, cmin, cmax = self.current_crop_bounds
        self.full_chlo_mask[:, rmin:rmax, cmin:cmax] = self.current_crop_chlo_mask

        prefix = f"{self.lif_combo.currentText()}_{self.vein_combo.currentText()}"
        out_dir = self.base_folder / "analysis" / "chlos_corrected"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{prefix}_chlo.tif"
        tifffile.imwrite(out_path, self.full_chlo_mask, imagej=True)
        self.unsaved_changes = False
        QMessageBox.information(
            self, "Saved", f"Corrected 3D masks saved to:\n{out_path}"
        )

    def export_data(self):
        if not self.base_folder:
            QMessageBox.warning(self, "Error", "No folder selected.")
            return

        # 1. Enforce saving before export
        self.apply_edits_to_master()  # Sync any active brush strokes
        if getattr(self, "unsaved_changes", False) or getattr(
            self, "mask_modified", False
        ):
            QMessageBox.warning(
                self,
                "Unsaved Changes",
                "You have unsaved edits in the current cell.\nPlease click 'Save Corrected Chlos (3D)' before exporting.",
            )
            return

        scope = self.combo_export_scope.currentText()
        resolved_only = self.chk_resolved_only.isChecked()
        use_microns = self.chk_microns.isChecked()
        export_chlo_rows = self.chk_export_chlo_rows.isChecked()

        export_dir = self.base_folder / "analysis" / "export"
        export_dir.mkdir(parents=True, exist_ok=True)

        # 2. Automated Export Paths & Filenames
        if scope == "Entire Folder":
            lif_names = list(self.lif_files.keys())
            file_name = f"{self.base_folder.name}.csv"
        elif scope == "Current LIF":
            if not self.current_lif:
                return
            lif_name_text = self.lif_combo.currentText()
            lif_names = [lif_name_text]
            stem = Path(lif_name_text).stem
            file_name = f"{stem}.csv"
        else:  # Current Vein
            if not self.current_lif or self.vein_combo.currentIndex() < 0:
                return
            lif_name_text = self.lif_combo.currentText()
            vein_name_current = self.vein_combo.currentText()
            lif_names = [lif_name_text]
            stem = Path(lif_name_text).stem
            file_name = f"{stem}-{vein_name_current}.csv"

        save_path = export_dir / file_name

        rows = []

        # --- Dynamic headers depending on row export mode ---
        if export_chlo_rows:
            headers = [
                "LIF_Name",
                "Vein_ID",
                "Cell_ID",
                f"Cell_Area_{'um2' if use_microns else 'px'}",
                f"Cell_Length_{'um' if use_microns else 'px'}",
                f"Cell_Width_{'um' if use_microns else 'px'}",
                "Num_Chloroplasts",
                "Occupancy",
                "Chloroplast_ID",  # ID column
                f"Chloroplast_Area_{'um2' if use_microns else 'px'}",
            ]
        else:
            headers = [
                "LIF_Name",
                "Vein_ID",
                "Cell_ID",
                f"Cell_Area_{'um2' if use_microns else 'px'}",
                f"Cell_Length_{'um' if use_microns else 'px'}",
                f"Cell_Width_{'um' if use_microns else 'px'}",
                "Num_Chloroplasts",
                "Occupancy",
                f"Chloroplast_Area(s)_{'um2' if use_microns else 'px'}",
            ]

        rows.append(headers)

        self.global_status_lbl.setText(
            f"Exporting {scope.lower()} to CSV... please wait."
        )
        QApplication.processEvents()  # Force UI to update before long computation

        px_to_um = 193.94 / 1024.0
        area_to_um2 = px_to_um**2

        for lif_name in lif_names:
            lif_path = self.lif_files[lif_name]
            lif_obj = LifFile(lif_path)

            for v_idx, img in enumerate(lif_obj.get_iter_image()):
                # Skip if we only want the current vein
                if scope == "Current Vein" and (
                    lif_name != self.lif_combo.currentText()
                    or v_idx != self.vein_combo.currentIndex()
                ):
                    continue

                vein_name = img.name
                prefix = f"{lif_name}_{vein_name}"

                is_active_vein = (
                    lif_name == self.lif_combo.currentText()
                    and v_idx == self.vein_combo.currentIndex()
                )

                if (
                    is_active_vein
                    and self.full_cell_mask is not None
                    and self.full_chlo_mask is not None
                ):
                    full_cell_mask_to_use = self.full_cell_mask
                    full_chlo_mask_to_use = self.full_chlo_mask
                else:
                    cell_path = (
                        self.base_folder / "analysis" / "cells" / f"{prefix}_cells.tif"
                    )
                    corrected_path = (
                        self.base_folder
                        / "analysis"
                        / "chlos_corrected"
                        / f"{prefix}_chlo.tif"
                    )
                    raw_path = (
                        self.base_folder / "analysis" / "chlos" / f"{prefix}_chlo.tif"
                    )

                    if not cell_path.exists():
                        continue

                    full_cell_mask_to_use = tifffile.imread(cell_path)

                    if corrected_path.exists():
                        full_chlo_mask_to_use = tifffile.imread(corrected_path)
                    elif raw_path.exists():
                        full_chlo_mask_to_use = tifffile.imread(raw_path)
                    else:
                        continue

                available_cells = np.unique(full_cell_mask_to_use)
                available_cells = available_cells[available_cells > 0]

                for cell_id in available_cells:

                    # --- NEW: Reset counter to 1 for EACH cell ---
                    cell_chlo_counter = 1

                    props = regionprops(
                        (full_cell_mask_to_use == cell_id).astype(np.uint8)
                    )
                    if not props:
                        continue

                    min_row, min_col, max_row, max_col = props[0].bbox
                    cell_length_px = max_row - min_row
                    cell_width_px = max_col - min_col
                    cell_area_px = props[0].area

                    pad = 20
                    rmin, rmax = max(0, min_row - pad), min(
                        full_cell_mask_to_use.shape[0], max_row + pad
                    )
                    cmin, cmax = max(0, min_col - pad), min(
                        full_cell_mask_to_use.shape[1], max_col + pad
                    )

                    crop_cell_mask = full_cell_mask_to_use[rmin:rmax, cmin:cmax]
                    crop_chlo_mask = full_chlo_mask_to_use[:, rmin:rmax, cmin:cmax]
                    target_cell_bool = crop_cell_mask == cell_id

                    rel, unrel = extract_all_chloroplasts_undirected(
                        target_cell_bool, crop_chlo_mask, iom_threshold=0.6
                    )

                    if resolved_only and len(unrel) > 0:
                        continue

                    chloro_areas_px = [c["peak_mask"].sum() for c in rel]
                    total_chloro_area_px = sum(chloro_areas_px)
                    occupancy = (
                        total_chloro_area_px / cell_area_px if cell_area_px > 0 else 0
                    )

                    if use_microns:
                        out_c_area = cell_area_px * area_to_um2
                        out_c_len = cell_length_px * px_to_um
                        out_c_wid = cell_width_px * px_to_um
                        out_ch_areas = [a * area_to_um2 for a in chloro_areas_px]
                    else:
                        out_c_area = cell_area_px
                        out_c_len = cell_length_px
                        out_c_wid = cell_width_px
                        out_ch_areas = chloro_areas_px

                    # --- ROW GENERATION LOGIC ---
                    c_area_fmt = f"{out_c_area:.2f}" if use_microns else out_c_area
                    c_len_fmt = f"{out_c_len:.2f}" if use_microns else out_c_len
                    c_wid_fmt = f"{out_c_wid:.2f}" if use_microns else out_c_wid
                    num_chlo = len(rel)
                    occ_fmt = f"{occupancy:.4f}"

                    if export_chlo_rows:
                        if len(out_ch_areas) == 0:
                            # Handle cells with 0 chloroplasts
                            rows.append(
                                [
                                    lif_name,
                                    vein_name,
                                    cell_id,
                                    c_area_fmt,
                                    c_len_fmt,
                                    c_wid_fmt,
                                    num_chlo,
                                    occ_fmt,
                                    "",  # Blank Chlo ID
                                    "",  # Blank Chlo Area
                                ]
                            )
                        else:
                            # Add a separate row for each chloroplast and increment counter
                            for ch_area in out_ch_areas:
                                rows.append(
                                    [
                                        lif_name,
                                        vein_name,
                                        cell_id,
                                        c_area_fmt,
                                        c_len_fmt,
                                        c_wid_fmt,
                                        num_chlo,
                                        occ_fmt,
                                        cell_chlo_counter,  # Use the cell-specific counter
                                        f"{ch_area:.2f}" if use_microns else ch_area,
                                    ]
                                )
                                cell_chlo_counter += 1
                    else:
                        # Fallback to the original semicolon-separated list
                        ch_areas_str = (
                            ";".join([f"{a:.2f}" for a in out_ch_areas])
                            if use_microns
                            else ";".join([str(a) for a in out_ch_areas])
                        )
                        rows.append(
                            [
                                lif_name,
                                vein_name,
                                cell_id,
                                c_area_fmt,
                                c_len_fmt,
                                c_wid_fmt,
                                num_chlo,
                                occ_fmt,
                                ch_areas_str,
                            ]
                        )

        try:
            with open(save_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            self.global_status_lbl.setText("Ready.")
            QMessageBox.information(
                self, "Export Complete", f"Data exported successfully to:\n{save_path}"
            )
        except Exception as e:
            self.global_status_lbl.setText("Export Failed.")
            QMessageBox.critical(self, "Export Error", f"Failed to save CSV:\n{str(e)}")
