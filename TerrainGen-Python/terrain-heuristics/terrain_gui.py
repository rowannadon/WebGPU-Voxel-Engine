# terrain_gui.py
# PyQt5 GUI wrapper for terrain_heuristics.py (your compute module).
# Place next to terrain_heuristics.py and run:  python terrain_gui.py

import os
import sys
import traceback
import numpy as np

from PIL import Image

from PyQt5.QtCore import Qt, QSize, pyqtSignal, QObject, QRunnable, QThreadPool, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QDockWidget, QFormLayout, QDoubleSpinBox,
    QSpinBox, QCheckBox, QGroupBox, QScrollArea, QTabWidget, QMessageBox,
    QLineEdit, QProgressBar, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)



# ---- Import your compute functions (unchanged) ----
# Make sure terrain_heuristics.py (your long script) sits alongside this file.
from terrain_heuristics import (
    load_heightmap,
    load_scalar_texture,
    compute_slope_aspect, compute_normals, compute_laplacian_curvature, compute_tpi,
    d8_flow_accumulation, compute_twi, compute_svf,
    compute_ocean_mask, compute_coastline_mask, distance_to_mask,
    latitude_degrees, prevailing_wind_3cell, directional_slope,
    temperature_from_lat_elev, precipitation_lat_bands, precipitation_orographic_advanced,
    potential_evapotranspiration, actual_evapotranspiration,
    classify_biomes_advanced, compute_foliage_color_rgb, compute_foliage_densities, BIOME_TABLE
)

# -----------------------
# Utilities (display)
# -----------------------

def _percentile_clip(arr: np.ndarray, lo=2, hi=98):
    a = np.asarray(arr, dtype=np.float64)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    plo, phi = np.percentile(a, [lo, hi])
    if phi <= plo: phi = plo + 1e-6
    a = (a - plo) / (phi - plo)
    return np.clip(a, 0, 1)

def scalar_to_qimage(arr: np.ndarray, lo=None, hi=None) -> QImage:
    a = np.asarray(arr, dtype=np.float64)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if lo is None or hi is None:
        a = _percentile_clip(a)
    else:
        if hi <= lo: hi = lo + 1e-6
        a = np.clip((a - lo) / (hi - lo), 0, 1)
    img8 = (a * 255.0 + 0.5).astype(np.uint8)
    h, w = img8.shape
    qimg = QImage(img8.data, w, h, w, QImage.Format_Grayscale8)
    return qimg.copy()

def rgb_to_qimage(rgb: np.ndarray) -> QImage:
    # rgb expected uint8 (H,W,3)
    arr = np.asarray(rgb, dtype=np.uint8, order="C")
    h, w, _ = arr.shape
    qimg = QImage(arr.data, w, h, 3*w, QImage.Format_RGB888)
    return qimg.copy()

def normal_to_qimage(n: np.ndarray) -> QImage:
    n = np.nan_to_num(n, nan=0.0)
    n = (np.clip(n, -1, 1) * 0.5 + 0.5)
    out = (n * 255.0 + 0.5).astype(np.uint8)
    return rgb_to_qimage(out)

# -----------------------
# TerrainEngine (caching)
# -----------------------

class TerrainEngine(QObject):
    """Holds data in memory and computes only what’s needed."""
    progress = pyqtSignal(str, int)      # message, %
    finished = pyqtSignal(dict, dict)    # images{key:QImage}, arrays{key:np.ndarray}
    failed = pyqtSignal(str)             # error text

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.path = None
        self.elev = None
        self.h = self.w = None
        self.lat1d = None

        # settings
        self.params = dict(
            cellsize=10.0, z_min=0.0, z_max=3000.0,
            sea_level_m=0.0,
            lapse_rate_c_per_km=6.5, t_equator_c=30.0, t_pole_c=0.0,
            coast_decay_km=1.75, orographic_alpha=4.0, shadow_beta=0.15,
            # Advanced rain shadow controls (match CLI defaults)
            shadow_max_distance_km=400.0,
            shadow_decay_km=150.0,
            shadow_height_threshold_m=150.0,
            shadow_strength=1.0,
            svf_dirs=16, svf_radius=100.0,
            tpi_radii=[25.0, 100.0],
            biome_mixing=1, use_random_biomes=False,
            flowacc_texture=None
        )
        # caches
        self.cache = {}
        # dependency buckets for invalidation
        self._dirty_all()

    # -------- Invalidation helpers --------
    def _dirty_all(self):
        self.cache.clear()

    def _dirty_core(self):
        # everything derived from elevation / cellsize
        keep = {}
        self.cache = keep  # drop all
    def _dirty_ocean_only(self):
        for k in ["ocean", "coastline", "d2coast_m"]:
            self.cache.pop(k, None)
        # Biome & foliage depend on ocean/coast distance
        for k in ["biome_id", "biome_rgb", "foliage_rgb", "P_mm", "AI", "AET", "PET", "temp_c"]:
            self.cache.pop(k, None)

    def _dirty_climate(self):
        for k in ["wind_u","wind_v","dir_s","P_mm","PET","AET","AI","temp_c","foliage_rgb",
                  "biome_id","biome_rgb"]:
            self.cache.pop(k, None)

    def _dirty_svf(self):
        for k in ["svf"]:
            self.cache.pop(k, None)

    def _dirty_tpi(self):
        # remove any tpi_* keys
        for k in list(self.cache.keys()):
            if k.startswith("tpi_"):
                self.cache.pop(k)

    def _dirty_biome_only(self):
        for k in ["biome_id","biome_rgb"]:
            self.cache.pop(k, None)

    def _dirty_flowacc(self):
        # Anything depending on flow accumulation or TWI
        for k in ["acc", "twi", "foliage_rgb", "biome_id", "biome_rgb",
                  "forest_density", "groundcover_density"]:
            self.cache.pop(k, None)

    # -------- Public API --------
    def set_settings(self, **kwargs):
        prev = self.params.copy()
        self.params.update(kwargs)

        # decide invalidation scope
        core_changed = any(prev[k] != self.params[k] for k in ["cellsize","z_min","z_max"])
        ocean_changed = (prev["sea_level_m"] != self.params["sea_level_m"])
        climate_changed = any(prev[k] != self.params[k] for k in
                              ["lapse_rate_c_per_km","t_equator_c","t_pole_c",
                               "coast_decay_km","orographic_alpha","shadow_beta",
                               "shadow_max_distance_km","shadow_decay_km",
                               "shadow_height_threshold_m","shadow_strength"])
        svf_changed = any(prev[k] != self.params[k] for k in ["svf_dirs","svf_radius"])
        tpi_changed = (prev["tpi_radii"] != self.params["tpi_radii"])
        biome_changed = any(prev[k] != self.params[k] for k in ["biome_mixing","use_random_biomes"])
        flowacc_changed = (prev.get("flowacc_texture") != self.params.get("flowacc_texture"))

        if core_changed:
            self._dirty_core()
        if ocean_changed:
            self._dirty_ocean_only()
        if climate_changed:
            self._dirty_climate()
        if svf_changed:
            self._dirty_svf()
        if tpi_changed:
            self._dirty_tpi()
        if biome_changed:
            self._dirty_biome_only()
        if flowacc_changed:
            self._dirty_flowacc()

    def load_heightmap_path(self, path):
        p = self.params
        self.path = path
        elev, in_bit_depth = load_heightmap(path, p["z_min"], p["z_max"])
        self.elev = elev.astype(np.float32)
        self.h, self.w = self.elev.shape
        self.lat1d = latitude_degrees(self.h)
        self._dirty_all()

    # ---- on-demand compute primitives ----
    def _need(self, key, fn):
        if key in self.cache: return self.cache[key]
        val = fn()
        self.cache[key] = val
        return val

    def get_ocean_masks(self):
        def _o():
            oce = compute_ocean_mask(self.elev, self.params["z_min"], self.params["z_max"], self.params["sea_level_m"])
            return oce
        ocean = self._need("ocean", _o)
        def _c():
            return compute_coastline_mask(ocean)
        coast = self._need("coastline", _c)
        def _d():
            d = distance_to_mask(coast, self.params["cellsize"])
            d[ocean] = 0.0
            return d
        d2 = self._need("d2coast_m", _d)
        return ocean, coast, d2

    def get_slope_aspect_normals(self):
        key = "slope_aspect"
        if key not in self.cache:
            s, a = compute_slope_aspect(self.elev, self.params["cellsize"])
            self.cache[key] = (s, a)
        slope_deg, aspect_deg = self.cache[key]
        normals = self._need("normals", lambda: compute_normals(self.elev, self.params["cellsize"]))
        # gradients for climate
        if "grad" not in self.cache:
            dzdx = np.gradient(self.elev, self.params["cellsize"], axis=1)
            dzdy = np.gradient(self.elev, self.params["cellsize"], axis=0)
            self.cache["grad"] = (dzdx.astype(np.float32), dzdy.astype(np.float32))
        return slope_deg, aspect_deg, normals

    def get_curvature(self):
        return self._need("curv", lambda: compute_laplacian_curvature(self.elev, self.params["cellsize"]))

    def get_tpi(self, r_m):
        tag = f"tpi_{int(r_m)}m"
        if tag in self.cache: return self.cache[tag]
        radius_px = max(1, int(round(r_m / self.params["cellsize"])))
        t = compute_tpi(self.elev, radius_px)
        self.cache[tag] = t
        return t

    def get_flowacc(self):
        def _acc():
            tex = self.params.get("flowacc_texture")
            if tex and os.path.exists(tex):
                # Load external texture as raw values, resize to elevation shape
                arr = load_scalar_texture(tex, target_shape=self.elev.shape)
                return arr.astype(np.float32)
            return d8_flow_accumulation(self.elev, self.params["cellsize"], resolve_pits='carve')
        return self._need("acc", _acc)

    def get_twi(self):
        slope_deg, _, _ = self.get_slope_aspect_normals()
        acc = self.get_flowacc()
        return self._need("twi", lambda: compute_twi(acc, slope_deg, self.params["cellsize"]))

    def get_svf(self):
        p = self.params
        return self._need("svf", lambda: compute_svf(self.elev, p["cellsize"], dirs=p["svf_dirs"], radius_m=p["svf_radius"]))

    def get_climate(self):
        p = self.params
        ocean, coast, d2coast = self.get_ocean_masks()
        slope_deg, aspect_deg, _ = self.get_slope_aspect_normals()
        dzdx, dzdy = self.cache["grad"]

        # winds
        u = self._need("wind_u", lambda: prevailing_wind_3cell(self.lat1d)[0])
        v = self._need("wind_v", lambda: prevailing_wind_3cell(self.lat1d)[1])

        dir_s = self._need("dir_s", lambda: directional_slope(dzdx, dzdy, u, v))
        temp = self._need("temp_c", lambda: temperature_from_lat_elev(self.lat1d, self.elev,
                                                                      p["lapse_rate_c_per_km"], p["t_equator_c"], p["t_pole_c"]))
        P = self._need("P_mm", lambda: precipitation_orographic_advanced(
            precipitation_lat_bands(self.lat1d), self.elev, u, v, dzdx, dzdy,
            d2coast, p["cellsize"], alpha=p["orographic_alpha"], beta=p["shadow_beta"],
            coast_decay_m=p["coast_decay_km"]*1000.0, coast_min_frac=0.35,
            # Advanced rain shadow tuning
            shadow_max_distance_km=p["shadow_max_distance_km"],
            shadow_decay_km=p["shadow_decay_km"],
            shadow_height_threshold_m=p["shadow_height_threshold_m"],
            shadow_strength=p["shadow_strength"]
        ))
        PET = self._need("PET", lambda: potential_evapotranspiration(temp, self.lat1d, k=20.0))
        AET = self._need("AET", lambda: actual_evapotranspiration(P, PET))
        AI = self._need("AI", lambda: P / (PET + 1e-6))

        return dict(ocean=ocean, coast=coast, d2coast=d2coast, dir_s=dir_s,
                    temp_c=temp, precip_mm=P, PET=PET, AET=AET, AI=AI,
                    slope_deg=slope_deg, aspect_deg=aspect_deg, u=u, v=v)

    def get_biome(self):
        cl = self.get_climate()
        twi = self.get_twi()
        tpi = self.get_tpi(50.0)
        p = self.params
        def _run():
            bid, brgb = classify_biomes_advanced(
                self.elev, p["sea_level_m"], cl["temp_c"], cl["precip_mm"], cl["PET"], twi,
                cl["slope_deg"], cl["aspect_deg"], tpi, cl["d2coast"], self.lat1d,
                cl["u"], cl["v"], mixing_radius=p["biome_mixing"],
                use_probabilistic=p["use_random_biomes"]
            )
            return bid, brgb
        if "biome_id" not in self.cache or "biome_rgb" not in self.cache:
            bid, brgb = _run()
            self.cache["biome_id"] = bid
            self.cache["biome_rgb"] = brgb
        return self.cache["biome_id"], self.cache["biome_rgb"]

    def get_foliage(self):
        cl = self.get_climate()
        svf = self.cache.get("svf", None)  # optional
        tpi_small = self.get_tpi(25.0)
        p = self.params
        return self._need("foliage_rgb", lambda: compute_foliage_color_rgb(
            elev=self.elev, ocean=cl["ocean"], temp_c=cl["temp_c"], precip_mm=cl["precip_mm"],
            pet_mm=cl["PET"], twi=self.cache.get("twi", None),
            slope_deg=cl["slope_deg"], aspect_deg=cl["aspect_deg"],
            dist_coast_m=cl["d2coast"], lat_deg_1d=self.lat1d, svf=svf,
            tpi_small=tpi_small, cellsize=p["cellsize"]
        ))

    def get_foliage_densities(self):
        cl = self.get_climate()
        svf = self.cache.get("svf", None)
        tpi_small = self.get_tpi(25.0)
        p = self.params
        key_f = "forest_density"
        key_g = "groundcover_density"
        if key_f in self.cache and key_g in self.cache:
            return self.cache[key_f], self.cache[key_g]
        fden, gden = compute_foliage_densities(
            elev=self.elev, ocean=cl["ocean"], temp_c=cl["temp_c"], precip_mm=cl["precip_mm"],
            pet_mm=cl["PET"], twi=self.cache.get("twi", None), slope_deg=cl["slope_deg"],
            aspect_deg=cl["aspect_deg"], dist_coast_m=cl["d2coast"], lat_deg_1d=self.lat1d,
            svf=svf, tpi_small=tpi_small, cellsize=p["cellsize"]
        )
        self.cache[key_f] = fden
        self.cache[key_g] = gden
        return fden, gden

    # ---- main entry used by Worker ----
    def compute_selected(self, selections):
        """selections: list of keys to compute"""
        if self.elev is None:
            raise RuntimeError("No heightmap loaded.")
        out_images = {}
        out_arrays = {}

        tot = len(selections)
        for idx, sel in enumerate(selections, 1):
            self.progress.emit(f"Computing {sel}...", int(100*idx/tot))

            if sel == "elevation":
                out_arrays["elevation"] = self.elev
                out_images["elevation"] = scalar_to_qimage(self.elev)

            elif sel == "ocean":
                ocean, coast, d2 = self.get_ocean_masks()
                out_arrays["ocean"] = ocean
                out_arrays["coastline"] = coast
                out_arrays["dist2coast_m"] = d2
                out_images["ocean"] = scalar_to_qimage(ocean.astype(np.float32), lo=0, hi=1)
                out_images["coastline"] = scalar_to_qimage(coast.astype(np.float32), lo=0, hi=1)
                out_images["dist2coast_m"] = scalar_to_qimage(d2)

            elif sel == "slope":
                slope_deg, aspect_deg, normals = self.get_slope_aspect_normals()
                out_arrays["slope_deg"] = slope_deg
                out_images["slope_deg"] = scalar_to_qimage(slope_deg, lo=0, hi=60)

            elif sel == "aspect":
                slope_deg, aspect_deg, normals = self.get_slope_aspect_normals()
                out_arrays["aspect_deg"] = aspect_deg
                out_images["aspect_deg"] = scalar_to_qimage(aspect_deg, lo=0, hi=360)

            elif sel == "normal":
                _, _, normals = self.get_slope_aspect_normals()
                out_arrays["normal"] = normals
                out_images["normal"] = normal_to_qimage(normals)

            elif sel == "curvature":
                curv = self.get_curvature()
                out_arrays["curvature"] = curv
                out_images["curvature"] = scalar_to_qimage(curv)

            elif sel.startswith("tpi"):
                # "tpi@25", "tpi@100"
                r_m = float(sel.split("@")[1])
                t = self.get_tpi(r_m)
                key = f"tpi_{int(r_m)}m"
                out_arrays[key] = t
                out_images[key] = scalar_to_qimage(t)

            elif sel == "flowacc":
                acc = self.get_flowacc()
                out_arrays["flowacc"] = acc
                out_images["flowacc_log"] = scalar_to_qimage(np.log1p(acc))

            elif sel == "twi":
                twi = self.get_twi()
                out_arrays["twi"] = twi
                out_images["twi"] = scalar_to_qimage(twi, lo=0, hi=15)

            elif sel == "svf":
                svf = self.get_svf()
                out_arrays["svf"] = svf
                out_images["svf"] = scalar_to_qimage(svf, lo=0, hi=1)

            elif sel == "climate":
                cl = self.get_climate()
                for k in ["temp_c","precip_mm","PET","AET","AI","dir_s"]:
                    out_arrays[k] = cl[k]
                out_images["temp_c"] = scalar_to_qimage(cl["temp_c"], lo=-30, hi=35)
                out_images["precip_mm"] = scalar_to_qimage(cl["precip_mm"])
                out_images["PET"] = scalar_to_qimage(cl["PET"])
                out_images["AET"] = scalar_to_qimage(cl["AET"])
                out_images["AI"] = scalar_to_qimage(cl["AI"], lo=0, hi=2)

            elif sel == "biome":
                bid, brgb = self.get_biome()
                out_arrays["biome_id"] = bid
                out_arrays["biome_rgb"] = brgb
                out_images["biome_id"] = scalar_to_qimage(bid, lo=0, hi=len(BIOME_TABLE)-1)
                out_images["biome_map"] = rgb_to_qimage(brgb)

            elif sel == "foliage":
                frgb = self.get_foliage()
                out_arrays["foliage_rgb"] = frgb
                out_images["foliage_color"] = rgb_to_qimage(frgb)

            elif sel == "forest_density":
                fden, gden = self.get_foliage_densities()
                out_arrays["forest_density"] = fden
                out_images["forest_density"] = scalar_to_qimage(fden, lo=0, hi=1)

            elif sel == "groundcover_density":
                fden, gden = self.get_foliage_densities()
                out_arrays["groundcover_density"] = gden
                out_images["groundcover_density"] = scalar_to_qimage(gden, lo=0, hi=1)

        self.finished.emit(out_images, out_arrays)

# -----------------------
# Worker for thread pool
# -----------------------

class WorkerSignals(QObject):
    progress = pyqtSignal(str, int)
    finished = pyqtSignal(object, object)
    failed = pyqtSignal(str)

class ComputeWorker(QRunnable):
    def __init__(self, engine: TerrainEngine, selections):
        super().__init__()
        self.engine = engine
        self.selections = selections
        self.signals = WorkerSignals()
        # bridge engine signals to worker signals (signal→signal)
        self.engine.progress.connect(self.signals.progress)
        self.engine.finished.connect(self.signals.finished)
        self.engine.failed.connect(self.signals.failed)

    def run(self):
        try:
            self.engine.compute_selected(self.selections)
        except Exception as e:
            tb = traceback.format_exc()
            self.signals.failed.emit(f"{e}\n\n{tb}")

# -----------------------
# UI
# -----------------------

class ZoomPanView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.pix = QGraphicsPixmapItem()
        self.scene().addItem(self.pix)

        # Smooth, mouse-centric zoom and hand panning
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        self._fit_on_set = True
        self._min_scale = 0.05
        self._max_scale = 40.0

    def set_image(self, qimg: QImage, fit=True):
        self.pix.setPixmap(QPixmap.fromImage(qimg))
        self.scene().setSceneRect(self.pix.boundingRect())
        self.resetTransform()
        self._fit_on_set = fit
        if fit:
            # If the view isn't laid out yet, defer fit to the next event loop tick
            if self.viewport().width() > 0 and self.viewport().height() > 0:
                self.fit_to_window()
            else:
                QTimer.singleShot(0, self.fit_to_window)

    def wheelEvent(self, ev):
        if self.pix.pixmap().isNull():
            return
        # standard wheel zoom
        delta = ev.angleDelta().y()
        if delta == 0:
            return
        # Zoom step (gentle)
        factor = 1.15 if delta > 0 else 1.0/1.15
        self._apply_zoom(factor)

    def keyPressEvent(self, ev):
        k = ev.key()
        if k in (Qt.Key_Plus, Qt.Key_Equal):      # + or =
            self._apply_zoom(1.15)
        elif k in (Qt.Key_Minus, Qt.Key_Underscore):
            self._apply_zoom(1.0/1.15)
        elif k in (Qt.Key_0, Qt.Key_ParenRight):  # 0 → 100%
            self.zoom_100()
        elif k in (Qt.Key_F,):
            self.fit_to_window()
        else:
            super().keyPressEvent(ev)

    def _apply_zoom(self, factor):
        # clamp overall scale
        cur = self.transform().m11()  # uniform scale assumed
        nxt = cur * factor
        if nxt < self._min_scale:
            factor = self._min_scale / max(cur, 1e-9)
        elif nxt > self._max_scale:
            factor = self._max_scale / max(cur, 1e-9)
        self.scale(factor, factor)
        self._fit_on_set = False  # user overrode fit

    def fit_to_window(self):
        if self.pix.pixmap().isNull():
            return
        self.resetTransform()
        self.fitInView(self.pix, Qt.KeepAspectRatio)
        self._fit_on_set = True 

    def zoom_100(self):
        self.resetTransform()
        self._fit_on_set = False

    # Keep auto-fit when view is shown or resized, until user zooms.
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit_on_set and not self.pix.pixmap().isNull():
            QTimer.singleShot(0, self.fit_to_window)

    def showEvent(self, event):
        super().showEvent(event)
        if self._fit_on_set and not self.pix.pixmap().isNull():
            QTimer.singleShot(0, self.fit_to_window)


class ImageTab(QWidget):
    def __init__(self, title: str, qimg: QImage):
        super().__init__()
        self.qimg = qimg

        layout = QVBoxLayout(self)

        # mini toolbar
        tb = QHBoxLayout()
        self.btn_fit = QPushButton("Fit")
        self.btn_100 = QPushButton("100%")
        self.btn_minus = QPushButton("-")
        self.btn_plus = QPushButton("+")
        tb.addWidget(self.btn_fit)
        tb.addWidget(self.btn_100)
        tb.addStretch(1)
        tb.addWidget(self.btn_minus)
        tb.addWidget(self.btn_plus)
        layout.addLayout(tb)

        # viewer
        self.view = ZoomPanView(self)
        layout.addWidget(self.view, 1)

        # wire buttons
        self.btn_fit.clicked.connect(self.view.fit_to_window)
        self.btn_100.clicked.connect(self.view.zoom_100)
        self.btn_plus.clicked.connect(lambda: self.view._apply_zoom(1.15))
        self.btn_minus.clicked.connect(lambda: self.view._apply_zoom(1.0/1.15))

        self.set_image(qimg)

    def set_image(self, qimg: QImage):
        self.qimg = qimg
        # Fit on first set for nicer default; user actions will detach fit
        self.view.set_image(qimg, fit=True)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Terrain Heuristics GUI")
        self.resize(1280, 800)

        self.engine = TerrainEngine()
        self.pool = QThreadPool.globalInstance()

        self._worker = None

        # Center: tabs with images
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        # Auto-fit when switching tabs if user hasn't overridden zoom
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Status
        self.status = self.statusBar()
        self.progress_label = QLabel("Idle")
        self.progress_label.setMinimumWidth(180)
        self.status.addPermanentWidget(self.progress_label, 0)
        self.progress = QProgressBar()
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        self.status.addPermanentWidget(self.progress, 1)

        # Left dock: settings (scrollable)
        dock = QDockWidget("Settings", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        dock.setWidget(scroll)
        dockw = QWidget()
        scroll.setWidget(dockw)
        outer = QVBoxLayout(dockw)

        # Load + general
        load_box = QGroupBox("Input & General")
        f = QFormLayout(load_box)
        self.path_label = QLabel("<i>No heightmap loaded</i>")
        btn_load = QPushButton("Load Heightmap")
        btn_load.clicked.connect(self.on_load)
        f.addRow(btn_load, self.path_label)

        # Optional external flow accumulation texture
        self.flowacc_path = QLineEdit("")
        self.flowacc_path.setPlaceholderText("Optional: use image instead of computing D8 flowacc")
        btn_flowacc = QPushButton("Browse…")
        btn_flowacc.clicked.connect(self.on_choose_flowacc)
        flowacc_row = QHBoxLayout()
        flowacc_row.addWidget(self.flowacc_path, 1)
        flowacc_row.addWidget(btn_flowacc)
        flowacc_row_w = QWidget()
        flowacc_row_w.setLayout(flowacc_row)
        f.addRow("Flowacc texture:", flowacc_row_w)

        self.cellsize = QDoubleSpinBox(); self.cellsize.setRange(0.1, 10000); self.cellsize.setValue(10.0); self.cellsize.setSuffix(" m/px")
        self.zmin = QDoubleSpinBox(); self.zmin.setRange(-10000, 10000); self.zmin.setValue(0.0); self.zmin.setSuffix(" m")
        self.zmax = QDoubleSpinBox(); self.zmax.setRange(-10000, 100000); self.zmax.setValue(3000.0); self.zmax.setSuffix(" m")
        self.sea = QDoubleSpinBox(); self.sea.setRange(-1000, 10000); self.sea.setValue(0.0); self.sea.setSuffix(" m")

        f.addRow("Cell size:", self.cellsize)
        f.addRow("Z min:", self.zmin)
        f.addRow("Z max:", self.zmax)
        f.addRow("Sea level:", self.sea)
        outer.addWidget(load_box)

        # Climate
        clim_box = QGroupBox("Climate / Orographic")
        cf = QFormLayout(clim_box)
        self.lapse = QDoubleSpinBox(); self.lapse.setRange(0, 15); self.lapse.setValue(6.5); self.lapse.setSuffix(" °C/km")
        self.teq = QDoubleSpinBox(); self.teq.setRange(-50, 60); self.teq.setValue(30.0); self.teq.setSuffix(" °C")
        self.tpole = QDoubleSpinBox(); self.tpole.setRange(-80, 30); self.tpole.setValue(0.0); self.tpole.setSuffix(" °C")
        self.coast_decay = QDoubleSpinBox(); self.coast_decay.setRange(0.1, 100); self.coast_decay.setValue(1.75); self.coast_decay.setSuffix(" km")
        self.alpha = QDoubleSpinBox(); self.alpha.setRange(0.0, 20.0); self.alpha.setValue(4.0)
        self.beta = QDoubleSpinBox(); self.beta.setRange(0.0, 5.0); self.beta.setValue(0.15)
        # Advanced rain shadow controls
        self.shadow_max_distance = QDoubleSpinBox(); self.shadow_max_distance.setRange(1.0, 10000.0); self.shadow_max_distance.setValue(400.0); self.shadow_max_distance.setSuffix(" km")
        self.shadow_decay = QDoubleSpinBox(); self.shadow_decay.setRange(1.0, 1000.0); self.shadow_decay.setValue(150.0); self.shadow_decay.setSuffix(" km")
        self.shadow_height_threshold = QDoubleSpinBox(); self.shadow_height_threshold.setRange(0.0, 5000.0); self.shadow_height_threshold.setValue(150.0); self.shadow_height_threshold.setSuffix(" m")
        self.shadow_strength = QDoubleSpinBox(); self.shadow_strength.setRange(0.0, 5.0); self.shadow_strength.setDecimals(2); self.shadow_strength.setSingleStep(0.05); self.shadow_strength.setValue(1.0)

        cf.addRow("Lapse rate:", self.lapse)
        cf.addRow("T@equator:", self.teq)
        cf.addRow("T@poles:", self.tpole)
        cf.addRow("Coast decay:", self.coast_decay)
        cf.addRow("Orographic:", self.alpha)
        cf.addRow("Shadow (beta, simple):", self.beta)
        # Advanced rain shadow tuning rows
        cf.addRow("Shadow max distance:", self.shadow_max_distance)
        cf.addRow("Shadow decay:", self.shadow_decay)
        cf.addRow("Shadow height threshold:", self.shadow_height_threshold)
        cf.addRow("Shadow strength:", self.shadow_strength)
        outer.addWidget(clim_box)

        # SVF / TPI
        svf_box = QGroupBox("SVF / TPI")
        sf = QFormLayout(svf_box)
        self.svf_dirs = QSpinBox(); self.svf_dirs.setRange(4, 128); self.svf_dirs.setValue(16)
        self.svf_radius = QDoubleSpinBox(); self.svf_radius.setRange(10, 5000); self.svf_radius.setValue(100.0); self.svf_radius.setSuffix(" m")
        self.tpi_radii = QLineEdit("25,100")  # meters, comma-separated

        sf.addRow("SVF dirs:", self.svf_dirs)
        sf.addRow("SVF radius:", self.svf_radius)
        sf.addRow("TPI radii (m):", self.tpi_radii)
        outer.addWidget(svf_box)

        # Biome opts
        bio_box = QGroupBox("Biomes")
        bf = QFormLayout(bio_box)
        self.biome_mix = QSpinBox(); self.biome_mix.setRange(0, 10); self.biome_mix.setValue(1)
        self.biome_rand = QCheckBox("Use probabilistic sampling")
        bf.addRow("Mixing radius:", self.biome_mix)
        bf.addRow(self.biome_rand)
        outer.addWidget(bio_box)

        # What to compute
        sel_box = QGroupBox("Generate")
        lay_sel = QVBoxLayout(sel_box)
        self.chk_elev = QCheckBox("Elevation"); self.chk_elev.setChecked(True)
        self.chk_ocean = QCheckBox("Ocean / Coast / Dist2Coast"); self.chk_ocean.setChecked(True)
        self.chk_slope = QCheckBox("Slope"); self.chk_slope.setChecked(True)
        self.chk_aspect = QCheckBox("Aspect")
        self.chk_normal = QCheckBox("Normal map")
        self.chk_curv = QCheckBox("Curvature")
        self.chk_tpi = QCheckBox("TPI (use radii above)")
        self.chk_acc = QCheckBox("Flow accumulation")
        self.chk_twi = QCheckBox("TWI")
        self.chk_svf = QCheckBox("SVF")
        self.chk_climate = QCheckBox("Climate (Temp, Precip, PET, AET, AI)"); self.chk_climate.setChecked(True)
        self.chk_biome = QCheckBox("Biome"); self.chk_biome.setChecked(True)
        self.chk_foliage = QCheckBox("Foliage color"); self.chk_foliage.setChecked(True)
        self.chk_forest = QCheckBox("Forest density")
        self.chk_ground = QCheckBox("Groundcover density")
        for w in [self.chk_elev,self.chk_ocean,self.chk_slope,self.chk_aspect,self.chk_normal,self.chk_curv,
                  self.chk_tpi,self.chk_acc,self.chk_twi,self.chk_svf,self.chk_climate,self.chk_biome,
                  self.chk_foliage,self.chk_forest,self.chk_ground]:
            lay_sel.addWidget(w)
        outer.addWidget(sel_box)

        # Actions
        btns = QHBoxLayout()
        self.btn_run = QPushButton("Generate")
        self.btn_run.clicked.connect(self.on_generate)
        self.btn_export = QPushButton("Export visible tabs to PNGs")
        self.btn_export.clicked.connect(self.on_export)
        self.btn_clear = QPushButton("Clear in-memory cache")
        self.btn_clear.clicked.connect(self.on_clear_cache)
        btns.addWidget(self.btn_run)
        btns.addWidget(self.btn_export)
        btns.addWidget(self.btn_clear)
        outer.addLayout(btns)

        outer.addStretch(1)

    # ----- UI Handlers -----
    def _collect_settings(self):
        # push settings into engine and mark dirties as needed
        tpistr = self.tpi_radii.text().strip()
        try:
            tpi_list = [float(x) for x in tpistr.replace(" ", "").split(",") if x]
        except:
            tpi_list = [25.0, 100.0]
        flowacc_tex = self.flowacc_path.text().strip() if hasattr(self, 'flowacc_path') else ''
        if flowacc_tex == "":
            flowacc_tex = None
        self.engine.set_settings(
            cellsize=self.cellsize.value(),
            z_min=self.zmin.value(),
            z_max=self.zmax.value(),
            sea_level_m=self.sea.value(),
            lapse_rate_c_per_km=self.lapse.value(),
            t_equator_c=self.teq.value(),
            t_pole_c=self.tpole.value(),
            coast_decay_km=self.coast_decay.value(),
            orographic_alpha=self.alpha.value(),
            shadow_beta=self.beta.value(),
            shadow_max_distance_km=self.shadow_max_distance.value(),
            shadow_decay_km=self.shadow_decay.value(),
            shadow_height_threshold_m=self.shadow_height_threshold.value(),
            shadow_strength=self.shadow_strength.value(),
            svf_dirs=self.svf_dirs.value(),
            svf_radius=self.svf_radius.value(),
            tpi_radii=tpi_list,
            biome_mixing=self.biome_mix.value(),
            use_random_biomes=self.biome_rand.isChecked(),
            flowacc_texture=flowacc_tex,
        )

    def _collect_selections(self):
        sel = []
        if self.chk_elev.isChecked(): sel.append("elevation")
        if self.chk_ocean.isChecked(): sel.append("ocean")
        if self.chk_slope.isChecked(): sel.append("slope")
        if self.chk_aspect.isChecked(): sel.append("aspect")
        if self.chk_normal.isChecked(): sel.append("normal")
        if self.chk_curv.isChecked(): sel.append("curvature")
        if self.chk_tpi.isChecked():
            for r in self.engine.params["tpi_radii"]:
                sel.append(f"tpi@{float(r)}")
        if self.chk_acc.isChecked(): sel.append("flowacc")
        if self.chk_twi.isChecked(): sel.append("twi")
        if self.chk_svf.isChecked(): sel.append("svf")
        if self.chk_climate.isChecked(): sel.append("climate")
        if self.chk_biome.isChecked(): sel.append("biome")
        if self.chk_foliage.isChecked(): sel.append("foliage")
        if self.chk_forest.isChecked(): sel.append("forest_density")
        if self.chk_ground.isChecked(): sel.append("groundcover_density")
        return sel

    def on_load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open heightmap PNG (8/16-bit)", "", "PNG Images (*.png);;All Files (*)")
        if not path: return
        try:
            self.engine.load_heightmap_path(path)
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))
            return
        self.path_label.setText(os.path.basename(path))
        self.status.showMessage(f"Loaded {path}  [{self.engine.w}-{self.engine.h}]", 5000)

    def on_generate(self):
        if self.engine.elev is None:
            QMessageBox.information(self, "No heightmap", "Load a heightmap first.")
            return
        self._collect_settings()
        selections = self._collect_selections()

        self.tabs.clear()
        self.progress.setValue(0)
        self.status.showMessage("Starting")
        self.progress_label.setText("Starting...")

        worker = ComputeWorker(self.engine, selections)
        worker.signals.progress.connect(self.on_progress)
        worker.signals.finished.connect(self.on_finished)
        worker.signals.failed.connect(self.on_failed)
        self.pool.start(worker)
        self._worker = worker

    def on_progress(self, msg, pct):
        self.status.showMessage(msg)
        self.progress.setValue(pct)
        self.progress_label.setText(msg)

    def on_finished(self, images: dict, arrays: dict):
        self.progress.setValue(100)
        self.status.showMessage("Done.", 3000)
        self.progress_label.setText("Idle")
        # populate tabs
        for name, qimg in images.items():
            tab = ImageTab(name, qimg)
            self.tabs.addTab(tab, name)
        if not images:
            QMessageBox.information(self, "Nothing generated", "No outputs were selected.")
        # Ensure the currently shown tab is fitted to the view
        self._on_tab_changed(self.tabs.currentIndex())

    def on_failed(self, txt):
        self.progress.setValue(0)
        self.progress_label.setText("Failed")
        QMessageBox.critical(self, "Computation failed", txt)

    def on_export(self):
        if self.tabs.count() == 0:
            QMessageBox.information(self, "No images", "Generate some maps first.")
            return
        outdir = QFileDialog.getExistingDirectory(self, "Choose export folder")
        if not outdir: return
        for i in range(self.tabs.count()):
            name = self.tabs.tabText(i)
            w: ImageTab = self.tabs.widget(i)
            qimg = w.qimg
            # save as PNG
            path = os.path.join(outdir, f"{name}.png")
            qimg.save(path, "PNG")
        self.status.showMessage(f"Exported to {outdir}", 5000)

    def on_clear_cache(self):
        self.engine._dirty_all()
        QMessageBox.information(self, "Cache cleared", "In-memory cache has been cleared.")

    def on_choose_flowacc(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose flow accumulation texture",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff);;All Files (*)"
        )
        if path:
            self.flowacc_path.setText(path)

    def _on_tab_changed(self, idx: int):
        w = self.tabs.widget(idx)
        if isinstance(w, ImageTab):
            # If still in auto-fit mode and image is present, fit now
            if w.view._fit_on_set and not w.view.pix.pixmap().isNull():
                QTimer.singleShot(0, w.view.fit_to_window)

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

