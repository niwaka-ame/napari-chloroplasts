import napari
import numpy as np
import tifffile
from pathlib import Path
from readlif.reader import LifFile
from skimage.morphology import flood
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QLabel,
    QFileDialog,
    QMessageBox,
    QCheckBox,
)


class CellSelectorWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.lif_files = {}
        self.current_lif = None
        self.base_folder = None

        self.cell_history = []

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
        # --- CELL SELECTOR UI ---
        # ==========================================
        self.layout.addWidget(QLabel("--- Editing Tools ---"))

        # Drawing Tools Row
        draw_layout = QHBoxLayout()
        self.draw_cb = QCheckBox("✎ Enable Drawing Mode")
        self.draw_undo_btn = QPushButton("⟲ Undo Draw")
        draw_layout.addWidget(self.draw_cb)
        draw_layout.addWidget(self.draw_undo_btn)
        self.layout.addLayout(draw_layout)

        self.layout.addWidget(
            QLabel("<i><b>Shift+Click:</b> Preview / Clear / Delete Cell</i>")
        )

        action_layout = QHBoxLayout()
        self.add_cell_btn = QPushButton("+ Add Cell (Spacebar)")
        self.undo_btn = QPushButton("⟲ Undo Last Added")
        action_layout.addWidget(self.add_cell_btn)
        action_layout.addWidget(self.undo_btn)
        self.layout.addLayout(action_layout)

        self.save_btn = QPushButton("💾 Save Cell Masks")
        self.layout.addWidget(self.save_btn)

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

        self.draw_cb.stateChanged.connect(self.toggle_drawing_mode)
        self.draw_undo_btn.clicked.connect(self.undo_drawing)

        self.add_cell_btn.clicked.connect(self.commit_preview)
        self.undo_btn.clicked.connect(self.undo_last_cell)
        self.save_btn.clicked.connect(self.save_cells)

        # --- View Events ---
        self.viewer.mouse_drag_callbacks.append(self.on_mouse_click)
        self.viewer.bind_key("Space", self.commit_preview_key)

    # --- Standard Interaction Logic ---
    def step_combo(self, combo: QComboBox, step: int):
        new_idx = combo.currentIndex() + step
        if 0 <= new_idx < combo.count():
            combo.setCurrentIndex(new_idx)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with LIF files")
        if folder:
            # 1. Save the REAL, full path to the internal variable
            self.base_folder = Path(folder)

            # 2. Create a truncated path for the UI that respects folder names
            max_len = 30
            if len(folder) > max_len:
                parts = self.base_folder.parts
                if len(parts) > 3:
                    # Format: Root.../ParentFolder/TargetFolder
                    display_text = f"{parts[0]}.../{parts[-2]}/{parts[-1]}"
                    # Fallback string slice if the last two folders are incredibly long
                    if len(display_text) > max_len:
                        display_text = folder[:20] + "..." + folder[-27:]
                else:
                    # Fallback for short depths with extremely long names
                    display_text = folder[:20] + "..." + folder[-27:]
            else:
                display_text = folder

            # 3. Apply the fake text to the label, but keep the real path in the tooltip
            self.folder_lbl.setText(display_text)
            self.folder_lbl.setToolTip(folder)

            self.load_btn.setEnabled(True)

    def load_data(self):
        if not self.base_folder or not self.base_folder.is_dir():
            return

        self.lif_files.clear()
        self.lif_combo.blockSignals(True)
        self.lif_combo.clear()

        for p in sorted(self.base_folder.glob("*.lif")):
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

        self.viewer.layers.clear()
        self.cell_history.clear()
        self.draw_cb.setChecked(False)

        # 1. Load Raw Data into a dictionary first
        channel_arrays = {}
        for c in range(min(c_dim, 3)):
            arr = np.zeros((z_dim, y_dim, x_dim), dtype=np.uint16)
            for z in range(z_dim):
                arr[z, :, :] = np.array(img.get_frame(z=z, c=c))
            channel_arrays[c] = arr

        # Add to viewer from bottom to top
        if 2 in channel_arrays:
            self.viewer.add_image(
                channel_arrays[2], name="Brightfield", colormap="gray", visible=False
            )

        if 1 in channel_arrays:
            self.viewer.add_image(
                channel_arrays[1],
                name="Raw Chlo",
                colormap="green",
                visible=True,
                blending="additive",
                opacity=0.6,
            )

        if 0 in channel_arrays:
            # Wall added last so it sits on top, with 0.75 opacity
            self.viewer.add_image(
                channel_arrays[0],
                name="Raw Wall",
                colormap="yellow",
                blending="additive",
                opacity=0.75,
            )

        # 2. Load Segmented Wall (Editable)
        prefix = f"{self.lif_combo.currentText()}_{self.vein_combo.currentText()}"
        wall_path = self.base_folder / "analysis" / "walls" / f"{prefix}_wall.tif"

        if wall_path.exists():
            wall_mask = tifffile.imread(wall_path)
        else:
            wall_mask = np.zeros((z_dim, y_dim, x_dim), dtype=np.uint8)

        self.viewer.add_labels(wall_mask, name="Editable Wall", opacity=1.0)

        # 3. Setup Cell Layers
        empty_mask = np.zeros((z_dim, y_dim, x_dim), dtype=np.uint16)
        self.viewer.add_labels(empty_mask.copy(), name="Cells Mask", opacity=0.6)

        preview_layer = self.viewer.add_labels(
            empty_mask.copy(), name="Preview", opacity=0.5
        )
        preview_layer.color_mode = "direct"
        preview_layer.color = {1: "red"}

        self.viewer.reset_view()

    # --- INTERACTION LOGIC ---
    def toggle_drawing_mode(self):
        if "Editable Wall" in self.viewer.layers:
            wall_layer = self.viewer.layers["Editable Wall"]
            if self.draw_cb.isChecked():
                self.viewer.layers.selection.active = wall_layer
                wall_layer.mode = "paint"
                wall_layer.brush_size = 1
                wall_layer.selected_label = 255
            else:
                wall_layer.mode = "pan_zoom"

    def undo_drawing(self):
        if "Editable Wall" in self.viewer.layers:
            self.viewer.layers["Editable Wall"].undo()

    def on_mouse_click(self, viewer, event):
        if "Shift" in event.modifiers and event.type == "mouse_press":
            if "Editable Wall" not in self.viewer.layers:
                return

            coords = np.round(self.viewer.cursor.position).astype(int)
            z, y, x = coords[0], coords[1], coords[2]

            wall_layer = self.viewer.layers["Editable Wall"]
            preview_layer = self.viewer.layers["Preview"]
            cells_layer = self.viewer.layers["Cells Mask"]

            shape = wall_layer.data.shape
            if not (0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]):
                return

            # Check what was clicked
            prev_val = preview_layer.data[z, y, x]
            cell_val = cells_layer.data[z, y, x]

            # 1. If clicking the active preview, clear it
            if prev_val == 1:
                preview_layer.data.fill(0)
                preview_layer.refresh()
                return

            # 2. If clicking an existing cell, delete it
            if cell_val > 0:
                cells_layer.data[cells_layer.data == cell_val] = 0
                cells_layer.refresh()
                if cell_val in self.cell_history:
                    self.cell_history.remove(cell_val)
                return

            # 3. Otherwise, if empty space, generate preview
            wall_slice = wall_layer.data[z]
            if wall_slice[y, x] != 0:
                return

            filled_mask = flood(wall_slice, (y, x), connectivity=1)

            preview_layer.data.fill(0)
            preview_layer.data[z][filled_mask] = 1
            preview_layer.refresh()

    def commit_preview_key(self, viewer):
        self.commit_preview()

    def commit_preview(self):
        if (
            "Preview" not in self.viewer.layers
            or "Cells Mask" not in self.viewer.layers
        ):
            return

        preview_layer = self.viewer.layers["Preview"]
        cells_layer = self.viewer.layers["Cells Mask"]

        # preview_idx contains (Z, Y, X) arrays of where the preview is 1
        preview_idx = np.where(preview_layer.data == 1)
        if len(preview_idx[0]) == 0:
            return

        new_id = cells_layer.data.max() + 1

        # Extract just the Y and X coordinates from the preview
        y_idx = preview_idx[1]
        x_idx = preview_idx[2]

        # Apply the new cell ID to ALL Z-slices (using the `:` slice)
        cells_layer.data[:, y_idx, x_idx] = new_id
        cells_layer.refresh()

        self.cell_history.append(new_id)

        # Clear preview
        preview_layer.data.fill(0)
        preview_layer.refresh()

    def undo_last_cell(self):
        if not self.cell_history or "Cells Mask" not in self.viewer.layers:
            return

        cells_layer = self.viewer.layers["Cells Mask"]
        last_id = self.cell_history.pop()

        cells_layer.data[cells_layer.data == last_id] = 0
        cells_layer.refresh()

    def save_cells(self):
        if "Cells Mask" not in self.viewer.layers:
            return

        cells_layer = self.viewer.layers["Cells Mask"]
        cells_mask_3d = cells_layer.data.astype(np.uint16)

        # Project 3D array down to 2D
        cells_mask_2d = np.max(cells_mask_3d, axis=0)

        prefix = f"{self.lif_combo.currentText()}_{self.vein_combo.currentText()}"
        out_dir = self.base_folder / "analysis" / "cells"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save 2D Cells Mask
        cell_path = out_dir / f"{prefix}_cells.tif"
        tifffile.imwrite(cell_path, cells_mask_2d, imagej=True)

        # Save 3D Editable Wall (so you don't lose drawing data)
        wall_layer = self.viewer.layers["Editable Wall"]
        wall_path = self.base_folder / "analysis" / "walls" / f"{prefix}_wall.tif"
        tifffile.imwrite(wall_path, wall_layer.data.astype(np.uint8), imagej=True)

        QMessageBox.information(self, "Saved", f"Saved 2D cells mask to {out_dir}")
