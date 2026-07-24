#!/usr/bin/env python3
"""
Mandala Mosaic Application
A simple PyQt5 application with a central canvas and side panels.
"""

import sys
import csv
import json
import math
import pickle
import random
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QFrame, QGridLayout, QLabel, QPushButton, QFileDialog, QCheckBox,
    QSpinBox, QLineEdit, QInputDialog, QMessageBox, QSizePolicy, QSlider,
    QColorDialog,
)
from PyQt5.QtCore import Qt, QPoint, QTimer, QBuffer, QByteArray, pyqtSignal
from PyQt5.QtGui import (
    QPainter, QColor, QPen, QPixmap, QBrush, QFont, QPolygon, QCursor,
    QLinearGradient, QImage,
)
import numpy as np


class Canvas(QWidget):
    """Central canvas widget for drawing/displaying content"""
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(600, 600)
        self.setStyleSheet("background-color: white; border: 1px solid black;")
        self.background_image = None
        # Snapshot of the just-loaded background pixmap at its initial
        # size — treated as the "100 %" reference for the sidebar's
        # Background Scale input. Every scale-spinbox change re-scales
        # from THIS reference, not from the current background_image,
        # so subsequent scale values don't compound.
        self.background_original = None
        
        # Polygon drawing mode variables
        self.polygon_mode = False
        self.polygon_points = []  # Points for the current polygon being drawn
        self.polygon_cursor_size = 10  # Size of the square cursor in pixels
        self.polygons = []  # List of completed polygons

        # Line drawing mode (mirrors duplicator.py). Click + drag to trace a
        # path; on release we place rotated squares of `line_polygon_size`
        # every `line_polygon_size + line_polygon_gap` world units along it.
        # If mandala_mode is on, each square gets num_copies radial copies.
        # Mutually exclusive with polygon_mode — set_line_mode / toggle_polygon_mode
        # keep them synchronised.
        self.line_mode = False
        self.is_drawing_line = False
        self.line_start_point = None
        self.current_line_end = None
        self.line_points = []                # world-coord path samples
        self.line_polygon_size = 15          # world-px side of each square
        self.line_polygon_gap = 2            # world-px gap between squares
        
        # Zoom and pan variables
        self.zoom_factor = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        self.is_panning = False
        self.last_pan_point = None
        
        # Image offset variables (separate from zoom/pan)
        self.image_offset_x = 0
        self.image_offset_y = 0
        
        # Radial copies setting
        self.num_copies = 6  # Default number of radial copies
        self.mandala_mode = True  # Whether to create radial copies (mandala mode)
        
        # Eraser mode
        self.eraser_mode = False
        self.is_erasing = False  # Track if currently dragging to erase
        
        # Fixed mandala center in world coordinates (will be set after widget is shown)
        self.mandala_center_world_x = None
        self.mandala_center_world_y = None
        
        # Center point offset
        self.center_offset_x = 0
        self.center_offset_y = 0
        
        # Parent shape tracking for polygon groups
        self.polygon_groups = []  # List of polygon groups, each group shares the same parent
        self.current_group_id = 0  # Counter for unique group IDs

        # Undo stack — full-state snapshots taken BEFORE each polygon-list
        # mutation. undo_last() pops and restores. Bounded to keep memory
        # sane on large sessions.
        self._undo_stack = []
        self._MAX_UNDO = 50
        
        # Selection tracking
        self.selected_polygon_index = -1  # Index of currently selected polygon (-1 means none)
        self.selected_polygon_indices = []  # List of all selected polygon indices (for group selection)
        
        # Control point editing
        self.selected_control_point = -1  # Index of selected control point (-1 means none)
        self.is_dragging_control_point = False
        self.control_point_size = 8  # Size of control point circles in pixels
        
        # Debug visualization
        self.debug_circle_dots = []  # List of (x, y) positions for debugging circular positions
        
        # Circle drawing
        self.show_circle = False
        self.circle_diameter = 1000          # Default inner-circle diameter
        self.outer_circle_diameter = 1500    # Default outer-circle diameter (drawn concentric with the inner one)
        
        # Circle drag handle
        self.is_dragging_center = False
        self.drag_handle_size = 12  # Size of the drag handle circle

        # Polygon body-drag state — set on left-click when a polygon is
        # selected, cleared on release. is_dragging_polygon distinguishes
        # a whole-polygon translate from is_dragging_control_point (single
        # vertex drag). _polygon_drag_last_world tracks the last cursor
        # world position so mouseMove computes incremental deltas.
        # _polygon_drag_did_snapshot ensures ONE undo checkpoint per drag.
        self.is_dragging_polygon = False
        self._polygon_drag_last_world = None
        self._polygon_drag_did_snapshot = False
        
        # Image drag handle
        self.is_dragging_image = False
        self.image_drag_start_offset_x = 0
        self.image_drag_start_offset_y = 0
        
        # Image visibility
        self.show_image = True  # Default to showing image

        # Color-tool modes (driven by the left panel's color palette).
        # paint_mode: a regular left-click on a polygon assigns the panel's
        # selected_color to it. eyedropper_mode: a left-click samples the
        # clicked polygon's color into the panel. replace_eyedropper_mode:
        # same as eyedropper but feeds the panel's "replace source" slot.
        self.paint_mode = False
        self.eyedropper_mode = False
        self.replace_eyedropper_mode = False
        self.background_color = QColor(255, 255, 255)
        self.left_panel = None
        self.right_panel = None
        
        # Enable mouse tracking for cursor display
        self.setMouseTracking(True)
        
        # Enable keyboard focus for key events
        self.setFocusPolicy(Qt.StrongFocus)
        
        # Timer for cursor updates in polygon mode
        self.cursor_timer = QTimer()
        self.cursor_timer.timeout.connect(self.update_cursor)
        # Don't start timer by default - only when entering polygon mode
    
    def update_cursor(self):
        """Update cursor display in polygon mode"""
        if self.polygon_mode:
            self.update()  # Refresh display to show cursor
    
    def screen_to_world(self, screen_x, screen_y):
        """Convert screen coordinates to world coordinates"""
        world_x = (screen_x - self.pan_offset_x) / self.zoom_factor
        world_y = (screen_y - self.pan_offset_y) / self.zoom_factor
        return world_x, world_y
    
    def world_to_screen(self, world_x, world_y):
        """Convert world coordinates to screen coordinates"""
        screen_x = world_x * self.zoom_factor + self.pan_offset_x
        screen_y = world_y * self.zoom_factor + self.pan_offset_y
        return screen_x, screen_y
    
    def set_num_copies(self, num_copies):
        """Set the number of radial copies to create"""
        self.num_copies = max(1, num_copies)  # Ensure at least 1 copy
    
    def set_mandala_mode(self, enabled):
        """Set whether to create radial copies (mandala mode)"""
        self.mandala_mode = enabled
    
    def set_eraser_mode(self, enabled):
        """Set whether eraser mode is enabled"""
        self.eraser_mode = enabled
        if enabled:
            # Exit polygon mode if active
            if self.polygon_mode:
                self.polygon_mode = False
                self.polygon_points = []  # Clear any in-progress polygon
                self.cursor_timer.stop()
                
                # Update polygon checkbox to reflect the change
                parent = self.parent()
                while parent:
                    for child in parent.findChildren(QCheckBox):
                        if child.text() == "Polygon" and hasattr(child, 'setChecked'):
                            child.blockSignals(True)
                            child.setChecked(False)
                            child.blockSignals(False)
                            break
                    parent = parent.parent()
            
            # Set cursor to indicate eraser mode
            self.setCursor(Qt.PointingHandCursor)
        else:
            # Reset cursor
            self.setCursor(Qt.ArrowCursor if not self.polygon_mode else Qt.BlankCursor)
        self.update()  # Refresh display
    
    def set_circle_visible(self, visible):
        """Set whether to show the circle"""
        self.show_circle = visible
        self.update()  # Refresh display
    
    def set_circle_diameter(self, diameter):
        """Set the circle diameter"""
        self.circle_diameter = max(1, diameter)  # Ensure positive value
        if self.show_circle:
            self.update()  # Refresh display if circle is visible

    def move_circles(self, dx: float, dy: float) -> None:
        """Nudge the mandala circles' centre by (dx, dy) in world units.
        Adjusts self.center_offset_x/y (the persistent offset added to
        the canvas-centre base position), then recomputes the mandala
        centre and repaints. Called by the sidebar arrow buttons."""
        self.center_offset_x += float(dx)
        self.center_offset_y += float(dy)
        self.update_mandala_center()
        self.update()

    def reset_circle_offset(self) -> None:
        """Reset the mandala circle centre back to the canvas centre.
        Zeros out center_offset_x/y and re-derives mandala_center_world_*."""
        self.center_offset_x = 0
        self.center_offset_y = 0
        self.update_mandala_center()
        self.update()

    def set_outer_circle_diameter(self, diameter):
        """Set the outer (concentric) circle diameter. Shown alongside the
        inner circle whenever 'Circle' is toggled on."""
        self.outer_circle_diameter = max(1, diameter)
        if self.show_circle:
            self.update()
    
    def set_image_visible(self, visible):
        """Set whether to show the background image"""
        self.show_image = visible
        self.update()  # Refresh display
    
    def get_circle_drag_handle_position(self):
        """Get the screen position of the circle drag handle (top-left of circle)"""
        if (not self.show_circle or 
            self.mandala_center_world_x is None or 
            self.mandala_center_world_y is None):
            return None, None
        
        # Get circle center in screen coordinates
        screen_center_x, screen_center_y = self.world_to_screen(
            self.mandala_center_world_x, 
            self.mandala_center_world_y
        )
        
        # Calculate circle radius in screen coordinates
        user_circle_radius_world = self.circle_diameter / 2.0
        circle_screen_radius = abs(user_circle_radius_world * self.zoom_factor)
        
        # Position handle at top-left of circle (45 degrees from center)
        import math
        angle = math.radians(225)  # 225 degrees = top-left
        handle_x = screen_center_x + circle_screen_radius * math.cos(angle)
        handle_y = screen_center_y + circle_screen_radius * math.sin(angle)
        
        return handle_x, handle_y
    
    def is_point_in_drag_handle(self, screen_x, screen_y):
        """Check if a screen point is inside the drag handle"""
        handle_x, handle_y = self.get_circle_drag_handle_position()
        if handle_x is None or handle_y is None:
            return False
        
        # Check if point is within handle circle
        distance = math.sqrt((screen_x - handle_x)**2 + (screen_y - handle_y)**2)
        return distance <= self.drag_handle_size / 2
    
    def get_image_drag_handle_position(self):
        """Get the screen position of the image drag handle (bottom-left of image)"""
        if (not self.show_image or 
            not self.background_image or 
            self.background_image.isNull()):
            return None, None
        
        # Get image position in world coordinates
        image_world_x = self.image_offset_x
        image_world_y = self.image_offset_y
        
        # Get image dimensions in world coordinates
        image_world_width = self.background_image.width()
        image_world_height = self.background_image.height()
        
        # Bottom-left corner of image in world coordinates
        bottom_left_world_x = image_world_x
        bottom_left_world_y = image_world_y + image_world_height
        
        # Convert to screen coordinates
        handle_x, handle_y = self.world_to_screen(bottom_left_world_x, bottom_left_world_y)
        
        return handle_x, handle_y
    
    def is_point_in_image_drag_handle(self, screen_x, screen_y):
        """Check if a screen point is inside the image drag handle"""
        handle_x, handle_y = self.get_image_drag_handle_position()
        if handle_x is None or handle_y is None:
            return False
        
        # Check if point is within handle circle
        distance = math.sqrt((screen_x - handle_x)**2 + (screen_y - handle_y)**2)
        return distance <= self.drag_handle_size / 2
    
    def set_background_image(self, image_path, desired_size=None):
        """Set background image for the canvas, optionally resizing it"""
        try:
            # Load the original image
            original_pixmap = QPixmap(image_path)
            
            if desired_size is not None:
                # Get original dimensions
                original_width = original_pixmap.width()
                original_height = original_pixmap.height()
                
                # Determine which side is longer
                longer_side = max(original_width, original_height)
                
                # Calculate scale factor
                scale_factor = desired_size / longer_side
                
                # Calculate new dimensions maintaining aspect ratio
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)
                
                # Resize the image
                self.background_image = original_pixmap.scaled(
                    new_width, new_height, 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
            else:
                # Use original size
                self.background_image = original_pixmap

            # Snapshot the freshly-loaded pixmap as the "100 %" reference
            # for the sidebar's Background Scale input. Every scale change
            # re-derives self.background_image from this snapshot so
            # values are always absolute (relative to first-load size),
            # not compounding on top of the current scale.
            self.background_original = self.background_image
            self.update()  # Trigger repaint
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False

    def set_background_scale(self, pct: float) -> None:
        """Rescale the background image to `pct` % of the ORIGINAL loaded
        size (i.e. the pixmap right after set_background_image cached
        it). Applied absolutely — passing 100 restores the original;
        passing 50 always halves the original, not the current size."""
        if (self.background_original is None
                or self.background_original.isNull()
                or pct <= 0):
            return
        ow = self.background_original.width()
        oh = self.background_original.height()
        if ow <= 0 or oh <= 0:
            return
        new_w = max(1, int(round(ow * float(pct) / 100.0)))
        new_h = max(1, int(round(oh * float(pct) / 100.0)))
        self.background_image = self.background_original.scaled(
            new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.update()
    
    def set_line_mode(self, enabled):
        """Enable / disable the line-drawing tool. Mutex with polygon_mode
        and eraser_mode — turning line mode on turns those off."""
        self.line_mode = bool(enabled)
        if self.line_mode:
            # Exit polygon mode + eraser mode so their handlers don't fire.
            if self.polygon_mode:
                self.polygon_mode = False
                self.polygon_points = []
                self.cursor_timer.stop()
                # Sync the "Polygon" checkbox in the right panel.
                parent = self.parent()
                while parent:
                    for child in parent.findChildren(QCheckBox):
                        if child.text() == "Polygon":
                            child.blockSignals(True)
                            child.setChecked(False)
                            child.blockSignals(False)
                            break
                    parent = parent.parent()
            if self.eraser_mode:
                self.eraser_mode = False
                parent = self.parent()
                while parent:
                    for child in parent.findChildren(QCheckBox):
                        if child.text() == "Eraser Mode":
                            child.blockSignals(True)
                            child.setChecked(False)
                            child.blockSignals(False)
                            break
                    parent = parent.parent()
            self.setCursor(Qt.CrossCursor)
        else:
            # Leaving line mode — clear any in-flight line drag.
            self.is_drawing_line = False
            self.line_points = []
            self.line_start_point = None
            self.current_line_end = None
            self.setCursor(Qt.ArrowCursor)
        self.update()

    def _snapshot_state(self):
        """Return a lightweight deep copy of the polygon state — enough to
        restore an undo without accidentally aliasing QColor / list refs
        with the live scene. polygon_groups' 'polygons' lists are dropped
        here and rebuilt on restore via group_id lookup, which avoids
        stale cross-references between the snapshot and the live data."""
        def copy_polygon(p):
            d = dict(p)
            if 'points' in d:
                d['points'] = list(d['points'])
            if isinstance(d.get('color'), QColor):
                d['color'] = QColor(d['color'])
            if isinstance(d.get('parent_shape'), dict):
                ps = dict(d['parent_shape'])
                if 'points' in ps:
                    ps['points'] = list(ps['points'])
                d['parent_shape'] = ps
            return d
        def copy_group(g):
            gd = dict(g)
            gd.pop('polygons', None)   # rebuilt on restore
            return gd
        return {
            'polygons': [copy_polygon(p) for p in self.polygons],
            'polygon_groups': [copy_group(g) for g in self.polygon_groups],
            'current_group_id': self.current_group_id,
        }

    def _push_undo_snapshot(self):
        """Save the current state to the undo stack. Call BEFORE any
        polygon-list mutation (add / remove / recolour / regroup)."""
        self._undo_stack.append(self._snapshot_state())
        # Bounded to _MAX_UNDO — drop the oldest snapshot when we overflow.
        if len(self._undo_stack) > self._MAX_UNDO:
            self._undo_stack.pop(0)

    def undo_last(self):
        """Restore the most recent snapshot. Returns True if something
        was undone, False if the stack was empty."""
        if not self._undo_stack:
            return False
        state = self._undo_stack.pop()
        self.polygons = state['polygons']
        self.polygon_groups = state['polygon_groups']
        self.current_group_id = state['current_group_id']
        # Re-link each group's 'polygons' list to the restored polygon dicts
        # by group_id (we dropped that list in _snapshot_state to keep the
        # snapshot cheap; here we rebuild it so downstream code that reads
        # group['polygons'] still works).
        for g in self.polygon_groups:
            gid = g.get('group_id')
            g['polygons'] = [
                p for p in self.polygons if p.get('group_id') == gid
            ]
        # Clear selection — the restored polygons have new object identity
        # so any cached index / reference is stale.
        self.selected_polygon_index = -1
        self.selected_polygon_indices = []
        self.selected_control_point = -1
        self.update()
        return True

    def set_line_polygon_size(self, size):
        """Set the size (world-px) of each square placed along the line."""
        self.line_polygon_size = max(1, int(size))

    @staticmethod
    def _qimage_to_numpy_rgb(qimg):
        """Convert a QImage OR QPixmap to an (H, W, 3) uint8 numpy array
        in RGB order. `background_image` in this app is a QPixmap (see
        set_background_image); accept it directly so the caller doesn't
        need to know."""
        if qimg is None:
            return None
        # QPixmap has no .format() — bounce through QImage first.
        if isinstance(qimg, QPixmap):
            qimg = qimg.toImage()
        if qimg.isNull():
            return None
        if qimg.format() != QImage.Format_RGB888:
            qimg = qimg.convertToFormat(QImage.Format_RGB888)
        w = qimg.width()
        h = qimg.height()
        stride = qimg.bytesPerLine()
        ptr = qimg.constBits()
        ptr.setsize(h * stride)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, stride)
        # bytesPerLine may include stride padding — trim to w*3 bytes per row.
        return arr[:, : w * 3].reshape(h, w, 3).copy()

    def detect_polygons_between_circles(self):
        """Detect polygons in the ring between the inner and outer circles
        using mosaic_to_csv.py's adaptive-threshold + connected-components
        + Douglas-Peucker pipeline. Each detected tile is added to
        self.polygons as a standalone polygon filled with its mean colour.

        Returns the number of polygons added."""
        if self.background_image is None or self.background_image.isNull():
            raise RuntimeError("No background image loaded.")

        # Lazy import so mosaic_to_csv is only loaded when Detect is clicked
        # (it pulls in ~a full Qt/PIL toolchain).
        import sys as _sys
        from pathlib import Path as _Path
        mtc_dir = (_Path(__file__).resolve().parent.parent
                   / "IMAGE_TO_MOSAIC" / "scripts")
        if str(mtc_dir) not in _sys.path:
            _sys.path.insert(0, str(mtc_dir))
        from mosaic_to_csv import detect_tiles

        # Convert background QImage → numpy RGB (image-pixel coords).
        img_rgb = self._qimage_to_numpy_rgb(self.background_image)
        if img_rgb is None:
            raise RuntimeError("Background image conversion failed.")
        h, w = img_rgb.shape[:2]

        # Circle geometry in image-pixel coords. World coords for the
        # background are offset by image_offset_x/y — subtract to reach
        # pixel space.
        self.initialize_mandala_center()
        cx_px = float(self.mandala_center_world_x) - float(self.image_offset_x)
        cy_px = float(self.mandala_center_world_y) - float(self.image_offset_y)
        r_in = float(self.circle_diameter) / 2.0
        r_out = float(self.outer_circle_diameter) / 2.0
        if r_in <= 0 or r_out <= r_in:
            raise RuntimeError(
                "Circle geometry is invalid — need inner < outer, both > 0.",
            )

        # Ring mask: True inside the ring. Apply to the input as black
        # outside the ring so adaptive-threshold + connected-components
        # can only find tiles between the two circles.
        Y, X = np.ogrid[:h, :w]
        dist_sq = (X - cx_px) ** 2 + (Y - cy_px) ** 2
        ring_mask = (dist_sq >= r_in ** 2) & (dist_sq <= r_out ** 2)
        masked = np.where(ring_mask[..., None], img_rgb, 0).astype(np.uint8)

        # Detection using the same defaults as mosaic_to_csv.py's GUI
        # (block_size=51, C=5, min_area=200, epsilon_ratio=0.02). Colours
        # sampled from the ORIGINAL background so tiles keep their real
        # colour instead of a mask-dimmed one.
        tiles = detect_tiles(
            masked,
            block_size=51, C=5, min_area=200, epsilon_ratio=0.02,
            color_source_rgb=img_rgb,
        )
        if not tiles:
            return 0

        # Undo checkpoint before adding.
        self._push_undo_snapshot()

        added = 0
        offx = float(self.image_offset_x)
        offy = float(self.image_offset_y)
        for pts, mean_rgb in tiles:
            # Defensive: centroid must be inside the ring — detect_tiles
            # already drops border-touching components, but a tile that
            # straddles the mask edge could still land partly outside.
            cxp = float(pts[:, 0].mean())
            cyp = float(pts[:, 1].mean())
            d_sq = (cxp - cx_px) ** 2 + (cyp - cy_px) ** 2
            if d_sq < r_in ** 2 or d_sq > r_out ** 2:
                continue

            r255 = max(0, min(255, int(round(mean_rgb[0] * 255))))
            g255 = max(0, min(255, int(round(mean_rgb[1] * 255))))
            b255 = max(0, min(255, int(round(mean_rgb[2] * 255))))
            polygon_data = {
                'points': [(float(p[0]) + offx, float(p[1]) + offy)
                           for p in pts],
                'color': QColor(r255, g255, b255),
                'is_single': True,
                'is_detected': True,
            }
            self.polygons.append(polygon_data)
            added += 1

        self.update()
        return added

    def set_line_polygon_gap(self, gap):
        """Set the gap (world-px) between consecutive squares."""
        self.line_polygon_gap = max(0, int(gap))

    @staticmethod
    def _rotated_square_points(cx, cy, size, angle_rad):
        """Return the 4 world-coord corners of a square centred at (cx, cy)
        with the given side length, rotated by angle_rad (0 = axis-aligned)."""
        half = size / 2.0
        local = [(-half, -half), (half, -half), (half, half), (-half, half)]
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        return [
            (cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a)
            for x, y in local
        ]

    def create_polygons_along_line(self):
        """Place squares along the traced line at (size + gap) intervals.
        Each square is aligned to the local path tangent. If mandala_mode
        is on, every square gets num_copies radial copies around the
        mandala centre; all squares (and their radial copies) share ONE
        group_id so eraser + regenerate treats them as one unit."""
        if len(self.line_points) < 2:
            return

        pts = list(self.line_points)
        # Cumulative segment lengths (world units).
        seg_lens = []
        total = 0.0
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            d = math.hypot(x2 - x1, y2 - y1)
            seg_lens.append(d)
            total += d
        if total <= 0:
            return

        step = self.line_polygon_size + self.line_polygon_gap
        if step <= 0:
            return
        n_polys = int(total / step)
        if n_polys <= 0:
            return

        def pos_and_angle_at(t):
            """Interpolate (x, y, tangent_angle) at path-distance t."""
            acc = 0.0
            for i, seg_len in enumerate(seg_lens):
                if seg_len <= 0:
                    continue
                if acc + seg_len >= t or i == len(seg_lens) - 1:
                    frac = min(1.0, max(0.0, (t - acc) / seg_len))
                    x1, y1 = pts[i]
                    x2, y2 = pts[i + 1]
                    cx = x1 + frac * (x2 - x1)
                    cy = y1 + frac * (y2 - y1)
                    ang = math.atan2(y2 - y1, x2 - x1)
                    return cx, cy, ang
                acc += seg_len
            # Fallback — last vertex.
            return pts[-1][0], pts[-1][1], 0.0

        # Fill colour for every square along the line = the palette
        # colour selected in the left panel. Falls back to opaque black
        # if the left panel isn't wired up (defensive).
        if (self.left_panel is not None
                and hasattr(self.left_panel, "selected_color")
                and isinstance(self.left_panel.selected_color, QColor)):
            fill_color = QColor(self.left_panel.selected_color)
        else:
            fill_color = QColor(0, 0, 0)

        # Build the base list of squares along the line (before any mandala
        # rotation). Each entry = (points, fill_color) — every square uses
        # the same palette colour.
        base_squares = []
        for i in range(n_polys):
            t = (i + 0.5) * step
            if t >= total:
                break
            cx, cy, ang = pos_and_angle_at(t)
            square = self._rotated_square_points(
                cx, cy, self.line_polygon_size, ang,
            )
            base_squares.append((square, QColor(fill_color)))

        if not base_squares:
            return

        # Undo checkpoint — captures state BEFORE the new line polygons +
        # their radial copies are added.
        self._push_undo_snapshot()

        # A single group_id ties the entire line (plus its radial copies)
        # together, so eraser + selection treat it as one editable unit.
        group_id = self.current_group_id
        self.current_group_id += 1
        group_polygons = []

        if self.mandala_mode:
            self.initialize_mandala_center()
            mcx = self.mandala_center_world_x
            mcy = self.mandala_center_world_y
            angle_step = 360.0 / self.num_copies if self.num_copies > 0 else 60.0
            for base_i, (square_pts, color) in enumerate(base_squares):
                # One parent_shape per base square, SHARED across its
                # radial copies (by object identity). This is what the
                # sibling-propagation code uses to find peers when a
                # single copy's control point is dragged.
                parent_shape = {
                    'is_line_base_square': True,
                    'base_index': base_i,
                    'num_copies': self.num_copies,
                }
                for copy_i in range(self.num_copies):
                    a_deg = copy_i * angle_step
                    a_rad = math.radians(a_deg)
                    cos_a, sin_a = math.cos(a_rad), math.sin(a_rad)
                    rotated = []
                    for wx, wy in square_pts:
                        rx = wx - mcx
                        ry = wy - mcy
                        rotated.append((
                            rx * cos_a - ry * sin_a + mcx,
                            rx * sin_a + ry * cos_a + mcy,
                        ))
                    polygon_data = {
                        'points': rotated,
                        'color': color,          # copies share the source square's colour
                        'group_id': group_id,
                        'copy_index': copy_i,
                        'rotation_angle': a_deg,
                        'is_line_polygon': True,
                        'parent_shape': parent_shape,  # shared across copies of this base square
                    }
                    self.polygons.append(polygon_data)
                    group_polygons.append(polygon_data)
        else:
            # No mandala — just drop each square as an independent polygon,
            # still tagged with the same group_id for easy bulk erase.
            for square_pts, color in base_squares:
                polygon_data = {
                    'points': square_pts,
                    'color': color,
                    'group_id': group_id,
                    'copy_index': 0,
                    'rotation_angle': 0.0,
                    'is_line_polygon': True,
                    'is_single': True,
                }
                self.polygons.append(polygon_data)
                group_polygons.append(polygon_data)

        # Record the group so downstream selection / regeneration logic
        # can find it. parent_shape is None because line-polygons are not
        # produced from a single parent polygon — reshape-all-copies is
        # not supported for line groups in this first version.
        self.polygon_groups.append({
            'group_id': group_id,
            'parent_shape': None,
            'polygons': group_polygons,
            'creation_time': len(self.polygon_groups),
            'is_line_group': True,
        })
        self.update()

    def toggle_polygon_mode(self):
        """Toggle polygon drawing mode on/off"""
        self.polygon_mode = not self.polygon_mode

        if self.polygon_mode:
            # Entering polygon mode — exit line-mode + eraser mode.
            if self.line_mode:
                self.set_line_mode(False)
                parent = self.parent()
                while parent:
                    for child in parent.findChildren(QCheckBox):
                        if child.text() == "Line":
                            child.blockSignals(True)
                            child.setChecked(False)
                            child.blockSignals(False)
                            break
                    parent = parent.parent()
            if self.eraser_mode:
                self.eraser_mode = False
                
                # Update eraser checkbox to reflect the change
                parent = self.parent()
                while parent:
                    for child in parent.findChildren(QCheckBox):
                        if child.text() == "Eraser Mode" and hasattr(child, 'setChecked'):
                            child.blockSignals(True)
                            child.setChecked(False)
                            child.blockSignals(False)
                            break
                    parent = parent.parent()
            
            self.polygon_points = []  # Reset points
            self.setCursor(Qt.BlankCursor)  # Hide cursor, we'll draw our own
            self.cursor_timer.start(50)  # Start cursor updates
        else:
            # Exiting polygon mode
            self.setCursor(Qt.ArrowCursor)  # Restore normal cursor
            self.polygon_points = []  # Clear any points
            self.cursor_timer.stop()  # Stop cursor updates
        
        self.update()  # Refresh display
    
    def add_polygon_point(self, screen_x, screen_y):
        """Add a point to the current polygon being drawn"""
        if not self.polygon_mode:
            return
        
        # Convert screen coordinates to world coordinates for storage
        world_x, world_y = self.screen_to_world(screen_x, screen_y)
        self.polygon_points.append((world_x, world_y))
        self.update()  # Refresh to show new point
    
    def finish_polygon(self):
        """Finish the current polygon if we have enough points"""
        if len(self.polygon_points) >= 3:
            if self.mandala_mode:
                # Create the original polygon plus radial copies
                self.create_radial_polygons()
            else:
                # Create only a single polygon (no copies)
                self.create_single_polygon()
        elif len(self.polygon_points) > 0:
            # Have some points but not enough
            # Keep the points, don't clear them - user might want to add more
            pass
        # If no points, do nothing silently
    
    def create_radial_polygons(self):
        """Create polygons arranged in a circle around center"""
        if len(self.polygon_points) < 3:
            return
        # Undo checkpoint — captures state BEFORE the new group is added.
        self._push_undo_snapshot()
        
        # Get mandala center in world coordinates (use fixed center)
        self.initialize_mandala_center()  # Ensure center is initialized
        center_world_x = self.mandala_center_world_x
        center_world_y = self.mandala_center_world_y
        
        # Calculate angle between each copy
        angle_step = 360.0 / self.num_copies if self.num_copies > 0 else 60.0
        
        # Create a new group for this set of polygons
        group_id = self.current_group_id
        self.current_group_id += 1
        
        # Store the original parent shape (first polygon in the group)
        parent_shape = {
            'points': list(self.polygon_points),  # Copy the original points
            'center': (center_world_x, center_world_y),
            'angle_step': angle_step,
            'num_copies': self.num_copies,
            'group_id': group_id,
            'creation_order': len(self.polygon_groups)
        }
        
        # Create group data structure
        group_polygons = []
        
        # Fill colour = the palette colour selected in the left panel.
        # Every polygon in this mandala group is filled with that colour
        # (falls back to opaque black if the left panel isn't wired up).
        if (self.left_panel is not None
                and hasattr(self.left_panel, "selected_color")
                and isinstance(self.left_panel.selected_color, QColor)):
            shared_color = QColor(self.left_panel.selected_color)
        else:
            shared_color = QColor(0, 0, 0)

        # Create specified number of polygons with calculated rotation
        for i in range(self.num_copies):
            angle_degrees = i * angle_step
            angle_radians = math.radians(angle_degrees)
            
            # Rotate each point around the center
            rotated_points = []
            for world_x, world_y in self.polygon_points:
                # Translate to origin (relative to center)
                rel_x = world_x - center_world_x
                rel_y = world_y - center_world_y
                
                # Apply rotation
                rotated_x = rel_x * math.cos(angle_radians) - rel_y * math.sin(angle_radians)
                rotated_y = rel_x * math.sin(angle_radians) + rel_y * math.cos(angle_radians)
                
                # Translate back
                final_x = rotated_x + center_world_x
                final_y = rotated_y + center_world_y
                
                rotated_points.append((final_x, final_y))
            
            # Create polygon data using the shared color from original points
            polygon_data = {
                'points': rotated_points,
                'color': shared_color,  # All copies use the same color
                'group_id': group_id,
                'copy_index': i,  # Index within the group (0 = original, 1+ = copies)
                'rotation_angle': angle_degrees,
                'parent_shape': parent_shape  # Reference to the parent shape data
            }
            
            self.polygons.append(polygon_data)
            group_polygons.append(polygon_data)
        
        # Store the group information
        group_info = {
            'group_id': group_id,
            'parent_shape': parent_shape,
            'polygons': group_polygons,
            'creation_time': len(self.polygon_groups)  # Simple timestamp
        }
        self.polygon_groups.append(group_info)
        
        # Clear current points after creating all polygons
        self.polygon_points = []
        self.update()  # Refresh display
    
    def create_single_polygon(self):
        """Create a single polygon without radial copies"""
        if len(self.polygon_points) < 3:
            return
        # Undo checkpoint — captures state BEFORE the new polygon lands.
        self._push_undo_snapshot()
        
        # Fill colour = the palette colour selected in the left panel.
        # Falls back to opaque black if the left panel isn't wired up.
        if (self.left_panel is not None
                and hasattr(self.left_panel, "selected_color")
                and isinstance(self.left_panel.selected_color, QColor)):
            color = QColor(self.left_panel.selected_color)
        else:
            color = QColor(0, 0, 0)
        
        # Create polygon data structure (similar to radial polygons but simpler)
        polygon_data = {
            'points': list(self.polygon_points),  # Copy the points
            'color': color,
            'is_single': True  # Mark as single polygon (not part of mandala group)
        }
        
        # Add to polygons list
        self.polygons.append(polygon_data)
        
        # Clear current points
        self.polygon_points = []
        self.update()  # Refresh display
    
    def get_polygon_group_by_id(self, group_id):
        """Get polygon group information by group ID"""
        for group in self.polygon_groups:
            if group['group_id'] == group_id:
                return group
        return None
    
    def get_parent_shape(self, polygon_data):
        """Get the parent shape for a given polygon"""
        if 'parent_shape' in polygon_data:
            return polygon_data['parent_shape']
        return None
    
    def get_siblings(self, polygon_data):
        """Get all sibling polygons (same parent) for a given polygon"""
        if 'group_id' not in polygon_data:
            return []
        
        group_info = self.get_polygon_group_by_id(polygon_data['group_id'])
        if group_info:
            return group_info['polygons']
        return []
    
    def regenerate_group(self, group_id, new_parent_points=None):
        """Regenerate all polygons in a group with optionally modified parent shape"""
        group_info = self.get_polygon_group_by_id(group_id)
        if not group_info:
            return False
        
        # Remove old polygons from the main list
        old_polygons = group_info['polygons']
        for old_poly in old_polygons:
            if old_poly in self.polygons:
                self.polygons.remove(old_poly)
        
        # Use new parent points if provided, otherwise use original
        if new_parent_points is None:
            new_parent_points = group_info['parent_shape']['points']
        else:
            # Update the parent shape with the new points
            group_info['parent_shape']['points'] = new_parent_points[:]
        
        # Temporarily set polygon_points to regenerate
        old_points = self.polygon_points
        old_num_copies = self.num_copies
        
        self.polygon_points = new_parent_points
        self.num_copies = group_info['parent_shape']['num_copies']
        
        # Create new polygons
        self.create_radial_polygons()
        
        # Restore original settings
        self.polygon_points = old_points
        self.num_copies = old_num_copies
        
        return True
    
    def get_average_color_from_background(self, world_points):
        """Get average color from background image at polygon area"""
        if not self.background_image or self.background_image.isNull():
            return QColor(128, 128, 128, 255)  # Default gray, fully opaque if no image
        
        try:
            # Convert QPixmap to QImage for pixel access
            background_image = self.background_image.toImage()
            
            # Convert world coordinates to image coordinates
            image_points = []
            for world_x, world_y in world_points:
                # Account for image offset
                image_x = world_x - self.image_offset_x
                image_y = world_y - self.image_offset_y
                image_points.append((int(image_x), int(image_y)))
            
            # Find bounding box of the polygon
            if not image_points:
                return QColor(128, 128, 128, 100)
            
            min_x = max(0, min(x for x, y in image_points))
            max_x = min(background_image.width() - 1, max(x for x, y in image_points))
            min_y = max(0, min(y for x, y in image_points))
            max_y = min(background_image.height() - 1, max(y for x, y in image_points))
            
            # Sample pixels within the polygon bounding box
            red_sum = green_sum = blue_sum = pixel_count = 0
            
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    # Simple point-in-polygon test using the bounding box
                    if self.point_in_polygon(x, y, image_points):
                        pixel = background_image.pixel(x, y)
                        red_sum += (pixel >> 16) & 0xFF
                        green_sum += (pixel >> 8) & 0xFF
                        blue_sum += pixel & 0xFF
                        pixel_count += 1
            
            if pixel_count > 0:
                # Calculate average color
                avg_red = red_sum // pixel_count
                avg_green = green_sum // pixel_count
                avg_blue = blue_sum // pixel_count
                return QColor(avg_red, avg_green, avg_blue, 255)  # Fully opaque
            else:
                return QColor(128, 128, 128, 255)  # Default gray, fully opaque
                
        except Exception as e:
            print(f"Error sampling background color: {e}")
            return QColor(128, 128, 128, 255)  # Default gray, fully opaque on error
    
    def point_in_polygon(self, x, y, polygon_points):
        """Simple point-in-polygon test using ray casting algorithm"""
        if len(polygon_points) < 3:
            return False
        
        inside = False
        j = len(polygon_points) - 1
        
        for i in range(len(polygon_points)):
            xi, yi = polygon_points[i]
            xj, yj = polygon_points[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside
    
    # ── Color-tool helpers (used by the left panel's color palette) ────────

    def find_polygon_at_world(self, world_x, world_y):
        """Return the index of the topmost polygon whose interior contains
        (world_x, world_y), or -1 if none. Topmost = last drawn (highest index)."""
        for i in range(len(self.polygons) - 1, -1, -1):
            pts = self.polygons[i].get('points') or []
            if len(pts) < 3:
                continue
            if self._point_in_polygon(world_x, world_y, pts):
                return i
        return -1

    @staticmethod
    def _point_in_polygon(x, y, pts):
        """Ray-cast point-in-polygon test."""
        inside = False
        n = len(pts)
        j = n - 1
        for i in range(n):
            xi, yi = pts[i]; xj, yj = pts[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
                inside = not inside
            j = i
        return inside

    def paint_polygon_at_point(self, world_x, world_y):
        """Assign the left panel's selected_color to the polygon at the click."""
        idx = self.find_polygon_at_world(world_x, world_y)
        if idx < 0:
            return
        selected = None
        if self.left_panel is not None and hasattr(self.left_panel, 'selected_color'):
            selected = self.left_panel.selected_color
        if selected is None:
            selected = QColor(0, 0, 0)
        new_color = QColor(selected)
        clicked = self.polygons[idx]
        # Undo checkpoint — captures the old colour(s) BEFORE repaint.
        self._push_undo_snapshot()
        # In mandala mode, repaint the entire group so all radial copies stay in sync.
        gid = clicked.get('group_id')
        if self.mandala_mode and gid is not None:
            for poly in self.polygons:
                if poly.get('group_id') == gid:
                    poly['color'] = QColor(new_color)
        else:
            clicked['color'] = new_color
        self.update()

    def sample_polygon_color_at_point(self, world_x, world_y):
        """Eyedropper: copy clicked polygon's color back to the left panel."""
        idx = self.find_polygon_at_world(world_x, world_y)
        self.eyedropper_mode = False
        self.setCursor(Qt.ArrowCursor)
        if idx < 0:
            return
        col = self.polygons[idx].get('color')
        if col is not None and self.left_panel is not None and hasattr(self.left_panel, 'receive_eyedropper_color'):
            self.left_panel.receive_eyedropper_color(QColor(col))

    def sample_replace_source_at_point(self, world_x, world_y):
        """Replace-source eyedropper: copy clicked polygon's color into the
        left panel's replace-from slot."""
        idx = self.find_polygon_at_world(world_x, world_y)
        self.replace_eyedropper_mode = False
        self.setCursor(Qt.ArrowCursor)
        if idx < 0:
            return
        col = self.polygons[idx].get('color')
        if col is not None and self.left_panel is not None and hasattr(self.left_panel, 'receive_replace_source_color'):
            self.left_panel.receive_replace_source_color(QColor(col))

    def mousePressEvent(self, event):
        """Handle mouse press events"""
        # Ensure canvas has focus for keyboard events
        self.setFocus()

        # Color-tool modes intercept clicks before any other behavior so the
        # user can sample / repaint a polygon regardless of polygon/eraser mode.
        if event.button() == Qt.LeftButton and (
            self.eyedropper_mode or self.replace_eyedropper_mode
        ):
            world_x, world_y = self.screen_to_world(event.x(), event.y())
            if self.replace_eyedropper_mode:
                self.sample_replace_source_at_point(world_x, world_y)
            else:
                self.sample_polygon_color_at_point(world_x, world_y)
            return
        # Right-click on a polygon (when not in polygon-draw mode) = repaint
        # with the left panel's selected color. Matches duplicator.py UX.
        if (event.button() == Qt.RightButton and not self.polygon_mode
                and self.left_panel is not None
                and hasattr(self.left_panel, 'selected_color')):
            world_x, world_y = self.screen_to_world(event.x(), event.y())
            self.paint_polygon_at_point(world_x, world_y)
            return

        # LEFT-click while Paint mode is ON = repaint the clicked polygon
        # with the palette colour. Wins over selection / drag / line /
        # polygon-draw so paint mode is unambiguous. In mandala mode the
        # entire group is repainted (paint_polygon_at_point handles that).
        if (event.button() == Qt.LeftButton and self.paint_mode
                and not self.polygon_mode and not self.line_mode
                and not self.eraser_mode):
            world_x, world_y = self.screen_to_world(event.x(), event.y())
            self.paint_polygon_at_point(world_x, world_y)
            return

        if event.button() == Qt.LeftButton and self.line_mode:
            # Line mode: press starts tracing a path.
            world_x, world_y = self.screen_to_world(event.x(), event.y())
            self.line_start_point = (world_x, world_y)
            self.current_line_end = (world_x, world_y)
            self.is_drawing_line = True
            self.line_points = [(world_x, world_y)]
            self.update()
            return
        if event.button() == Qt.LeftButton and self.polygon_mode:
            # In polygon mode, left click adds point to polygon
            self.add_polygon_point(event.x(), event.y())
        elif event.button() == Qt.RightButton and self.polygon_mode:
            # Right click finishes the polygon
            self.finish_polygon()
        elif event.button() == Qt.LeftButton and not self.polygon_mode:
            # Check for eraser mode first
            if self.eraser_mode:
                world_x, world_y = self.screen_to_world(event.x(), event.y())
                self.erase_polygon_at_point(world_x, world_y)
                self.is_erasing = True  # Start erasing mode for drag
                return
            
            # Check for drag handle click
            if self.is_point_in_drag_handle(event.x(), event.y()):
                # Start dragging center point
                self.is_dragging_center = True
                self.setCursor(Qt.ClosedHandCursor)
                return
            elif self.is_point_in_image_drag_handle(event.x(), event.y()):
                # Start dragging image
                self.is_dragging_image = True
                self.image_drag_start_offset_x = self.image_offset_x
                self.image_drag_start_offset_y = self.image_offset_y
                self.last_pan_point = event.pos()
                self.setCursor(Qt.ClosedHandCursor)
                return
            
            # In selection mode, check for control point clicks
            control_point_index = self.find_control_point_at_screen_pos(event.x(), event.y())
            
            if control_point_index >= 0:
                # Start dragging control point
                self.selected_control_point = control_point_index
                self.is_dragging_control_point = True
                self.setCursor(Qt.ClosedHandCursor)
                self.update()
            else:
                # Polygon selection + body-drag. Matches duplicator.py's UX:
                # first click on a polygon SELECTS it; a subsequent click
                # INSIDE the already-selected polygon starts a body-drag.
                # This split-click pattern avoids the "every click triggers
                # a drag" surprise.
                world_x, world_y = self.screen_to_world(event.x(), event.y())
                if (self.selected_polygon_index is not None
                        and self.selected_polygon_index >= 0
                        and self.selected_polygon_index < len(self.polygons)
                        and self.point_in_polygon(
                            world_x, world_y,
                            self.polygons[self.selected_polygon_index]["points"])):
                    # Start dragging the already-selected polygon.
                    self.is_dragging_polygon = True
                    self._polygon_drag_last_world = (world_x, world_y)
                    self._polygon_drag_did_snapshot = False
                    self.setCursor(Qt.ClosedHandCursor)
                else:
                    # First click on a polygon (or click on empty space).
                    # select_polygon_at_point sets selected_polygon_index
                    # to the hit polygon or -1 if nothing was hit.
                    self.select_polygon_at_point(world_x, world_y)
        elif event.button() == Qt.MiddleButton:
            # Start panning
            self.is_panning = True
            self.last_pan_point = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events"""
        if self.is_drawing_line:
            # Line mode — append a path sample if it's far enough from the
            # last one. Threshold in world units matches duplicator.py so
            # zoom doesn't over/undersample the path.
            world_x, world_y = self.screen_to_world(event.x(), event.y())
            self.current_line_end = (world_x, world_y)
            if not self.line_points:
                self.line_points.append((world_x, world_y))
            else:
                lx, ly = self.line_points[-1]
                if math.hypot(world_x - lx, world_y - ly) > 5:
                    self.line_points.append((world_x, world_y))
            self.update()
            return
        if self.is_erasing:
            # Continue erasing while dragging
            world_x, world_y = self.screen_to_world(event.x(), event.y())
            self.erase_polygon_at_point(world_x, world_y)
        elif self.is_dragging_center:
            # Drag center point by converting screen movement to world offset
            world_x, world_y = self.screen_to_world(event.x(), event.y())
            
            # Calculate base center position (canvas center without offset)
            canvas_center_screen_x = self.width() / 2
            canvas_center_screen_y = self.height() / 2
            base_center_x, base_center_y = self.screen_to_world(canvas_center_screen_x, canvas_center_screen_y)
            
            # Calculate new offset from dragged position
            new_offset_x = world_x - base_center_x
            new_offset_y = world_y - base_center_y
            
            # Update center offset and mandala center
            self.center_offset_x = new_offset_x
            self.center_offset_y = new_offset_y
            self.update_mandala_center()
            self.update()
            
        elif self.is_dragging_image:
            # Drag image by converting screen movement to world offset
            if self.last_pan_point:
                delta = event.pos() - self.last_pan_point
                # Convert screen delta to world delta (accounting for zoom)
                world_delta_x = delta.x() / self.zoom_factor
                world_delta_y = delta.y() / self.zoom_factor
                
                # Update image offset
                self.image_offset_x += world_delta_x
                self.image_offset_y += world_delta_y
                self.last_pan_point = event.pos()
                self.update()
            
        elif self.is_dragging_polygon and self._polygon_drag_last_world is not None:
            # Whole-polygon translate: apply the world-space delta since
            # the last mouse position, propagating to mandala siblings if
            # applicable. Undo checkpoint is taken on the first actual
            # movement (not on plain click-release with no drag).
            world_x, world_y = self.screen_to_world(event.x(), event.y())
            lx, ly = self._polygon_drag_last_world
            dx = world_x - lx
            dy = world_y - ly
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                if not self._polygon_drag_did_snapshot:
                    self._push_undo_snapshot()
                    self._polygon_drag_did_snapshot = True
                self._translate_selected_polygon(dx, dy)
                self._polygon_drag_last_world = (world_x, world_y)
        elif self.is_dragging_control_point and self.selected_control_point >= 0:
            # Drag control point to reshape polygon
            world_x, world_y = self.screen_to_world(event.x(), event.y())
            
            # Update only the control point position of the primary selected polygon
            if (self.selected_polygon_index >= 0 and 
                self.selected_polygon_index < len(self.polygons)):
                
                polygon_data = self.polygons[self.selected_polygon_index]
                points = polygon_data['points']
                
                if self.selected_control_point < len(points):
                    # Update only the selected control point in the primary polygon
                    points[self.selected_control_point] = (world_x, world_y)
                    self.update()
                    
        elif self.is_panning and self.last_pan_point:
            # Update pan offset
            delta = event.pos() - self.last_pan_point
            self.pan_offset_x += delta.x()
            self.pan_offset_y += delta.y()
            self.last_pan_point = event.pos()
            self.update()
        elif self.polygon_mode:
            # Update cursor position for polygon mode
            self.update()
        else:
            # Check if hovering over drag handles and update cursor
            if (not self.is_dragging_control_point and 
                not self.is_panning):
                
                if self.is_point_in_drag_handle(event.x(), event.y()):
                    self.setCursor(Qt.OpenHandCursor)
                elif (self.background_image and 
                      self.is_point_in_image_drag_handle(event.x(), event.y())):
                    self.setCursor(Qt.OpenHandCursor)
                elif not self.polygon_mode and not self.eraser_mode:
                    self.setCursor(Qt.ArrowCursor)
                elif self.eraser_mode and not self.polygon_mode:
                    self.setCursor(Qt.PointingHandCursor)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if self.is_drawing_line:
            # Line mode — bake the traced path into polygons.
            self.create_polygons_along_line()
            self.is_drawing_line = False
            self.line_points = []
            self.line_start_point = None
            self.current_line_end = None
            self.update()
            return
        if self.is_erasing:
            # Stop erasing
            self.is_erasing = False
        elif self.is_dragging_center:
            # Stop dragging center
            self.is_dragging_center = False
            self.setCursor(Qt.ArrowCursor if not self.polygon_mode else Qt.BlankCursor)
        elif self.is_dragging_polygon:
            # End polygon body-drag. Undo snapshot was pushed on the
            # first movement; nothing else to persist here.
            self.is_dragging_polygon = False
            self._polygon_drag_last_world = None
            self._polygon_drag_did_snapshot = False
            self.setCursor(Qt.ArrowCursor)
        elif self.is_dragging_control_point:
            # When control-point dragging finishes in mandala mode, mirror
            # the move into every sibling copy (same parent_shape by
            # reference). The propagation walks siblings itself — we don't
            # need the whole group to be selected any more (selection is
            # one polygon at a time now).
            if (self.mandala_mode
                    and self.selected_polygon_index >= 0
                    and self.selected_control_point >= 0):
                polygon_data = self.polygons[self.selected_polygon_index]
                points = polygon_data['points']
                if self.selected_control_point < len(points):
                    final_x, final_y = points[self.selected_control_point]
                    self.update_corresponding_points_in_copies(final_x, final_y)
                    self.update()  # force a redraw
            # Stop dragging control point
            self.is_dragging_control_point = False
            self.selected_control_point = -1
            self.setCursor(Qt.ArrowCursor)
        elif self.is_dragging_image:
            self.is_dragging_image = False
            self.last_pan_point = None
            self.setCursor(Qt.ArrowCursor)
        elif self.is_panning:
            self.is_panning = False
            self.last_pan_point = None
            self.setCursor(Qt.ArrowCursor if not self.polygon_mode else Qt.BlankCursor)
    
    def wheelEvent(self, event):
        """Wheel behaviour matches duplicator.py:
          - If a polygon is SELECTED → rotate it (and mandala siblings)
            by 5° per wheel notch, direction from angleDelta sign.
          - Otherwise → the existing cursor-anchored zoom.
        Deselect (Esc or click empty space) to zoom while a polygon is
        selected."""
        # Rotate the selected polygon (+ mandala siblings). No cursor-
        # inside check — as long as a polygon is selected, wheel rotates.
        idx = self.selected_polygon_index
        if idx is not None and 0 <= idx < len(self.polygons):
            self._push_undo_snapshot()
            step_deg = 5.0 if event.angleDelta().y() > 0 else -5.0
            self._rotate_selected_polygon(math.radians(step_deg))
            event.accept()
            return

        # Fall through: cursor-anchored zoom.
        mouse_pos = event.pos()
        old_world_x, old_world_y = self.screen_to_world(
            mouse_pos.x(), mouse_pos.y(),
        )
        zoom_in = event.angleDelta().y() > 0
        zoom_factor = 1.25 if zoom_in else 0.8
        new_zoom = self.zoom_factor * zoom_factor
        if 0.1 <= new_zoom <= 10.0:
            self.zoom_factor = new_zoom
            new_screen_x, new_screen_y = self.world_to_screen(
                old_world_x, old_world_y,
            )
            self.pan_offset_x += mouse_pos.x() - new_screen_x
            self.pan_offset_y += mouse_pos.y() - new_screen_y
            self.update()
    
    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key_Escape:
            # Escape key releases polygon mode
            if self.polygon_mode:
                self.polygon_mode = False
                self.polygon_points = []  # Clear any in-progress polygon
                self.setCursor(Qt.ArrowCursor)
                self.cursor_timer.stop()
                self.update()
                
                # Find and update the checkbox - look for it in the widget hierarchy
                parent = self.parent()
                while parent:
                    # Look for SidePanel widgets in the parent's children
                    for child in parent.findChildren(QCheckBox):
                        if child.text() == "Polygon" and hasattr(child, 'setChecked'):
                            # Block signals to prevent triggering toggle_polygon_mode again
                            child.blockSignals(True)
                            child.setChecked(False)
                            child.blockSignals(False)
                            break
                    parent = parent.parent()
                    
        elif event.key() == Qt.Key_P:
            # P key toggles polygon mode
            self.toggle_polygon_mode()
            
            # Find and update the checkbox - look for it in the widget hierarchy
            parent = self.parent()
            while parent:
                # Look for SidePanel widgets in the parent's children
                for child in parent.findChildren(QCheckBox):
                    if child.text() == "Polygon" and hasattr(child, 'setChecked'):
                        # Block signals to prevent triggering toggle_polygon_mode again
                        child.blockSignals(True)
                        child.setChecked(self.polygon_mode)
                        child.blockSignals(False)
                        break
                parent = parent.parent()
                
        elif event.key() == Qt.Key_E:
            # E key toggles eraser mode
            self.set_eraser_mode(not self.eraser_mode)
            
            # Find and update the checkbox - look for it in the widget hierarchy
            parent = self.parent()
            while parent:
                # Look for SidePanel widgets in the parent's children
                for child in parent.findChildren(QCheckBox):
                    if child.text() == "Eraser Mode" and hasattr(child, 'setChecked'):
                        # Block signals to prevent triggering toggle_polygon_mode again
                        child.blockSignals(True)
                        child.setChecked(self.eraser_mode)
                        child.blockSignals(False)
                        break
                parent = parent.parent()
                    
        elif event.key() == Qt.Key_Delete:
            # Delete key removes selected polygon(s)
            if self.selected_polygon_indices:
                self.delete_selected_polygon()
        else:
            super().keyPressEvent(event)
    
    def delete_selected_polygon(self):
        """Delete the currently selected polygon(s) and update groups"""
        if not self.selected_polygon_indices:
            return
        
        # Sort indices in descending order to avoid index shifting issues when deleting
        indices_to_delete = sorted(self.selected_polygon_indices, reverse=True)
        
        # Delete polygons and track affected groups
        affected_groups = set()
        
        for index in indices_to_delete:
            if 0 <= index < len(self.polygons):
                polygon_to_delete = self.polygons[index]
                
                # Track which groups are affected
                if 'group_id' in polygon_to_delete:
                    affected_groups.add(polygon_to_delete['group_id'])
                
                # Remove from polygons list
                self.polygons.pop(index)
        
        # Update polygon groups for affected groups
        for group_id in affected_groups:
            for group in self.polygon_groups[:]:  # Use slice copy to avoid modification during iteration
                if group['group_id'] == group_id:
                    # Rebuild the group's polygon list
                    group['polygons'] = [p for p in group['polygons'] if p in self.polygons]
                    # If group is now empty, remove it
                    if not group['polygons']:
                        self.polygon_groups.remove(group)
        
        # Clear selection
        self.selected_polygon_index = -1
        self.selected_polygon_indices = []
        self.update()
    
    def erase_polygon_at_point(self, world_x, world_y):
        """Erase the specific polygon at the given point (not its copies)"""
        # Find polygon at point
        for i, polygon_data in enumerate(self.polygons):
            points = polygon_data['points']
            if self.point_in_polygon(world_x, world_y, points):
                # Undo checkpoint — captures the polygon BEFORE removal.
                self._push_undo_snapshot()
                # Remove this specific polygon
                affected_group_id = polygon_data.get('group_id')
                self.polygons.pop(i)
                
                # Update polygon groups
                if affected_group_id is not None:
                    for group in self.polygon_groups[:]:
                        if group['group_id'] == affected_group_id:
                            # Rebuild the group's polygon list
                            group['polygons'] = [p for p in group['polygons'] if p in self.polygons]
                            # If group is now empty, remove it
                            if not group['polygons']:
                                self.polygon_groups.remove(group)
                
                self.update()
                return True  # Successfully erased one polygon
        return False  # No polygon found at point
    
    def select_polygon_at_point(self, world_x, world_y):
        """Select a single polygon at the given world coordinates. Even in
        mandala mode, selection is one-polygon-at-a-time — but dragging a
        control point still propagates to sibling copies via
        update_corresponding_points_in_copies (siblings identified by
        shared parent_shape reference + their rotation_angle)."""
        self.selected_polygon_index = -1
        self.selected_polygon_indices = []

        # Check polygons in reverse order (last drawn first).
        for i in range(len(self.polygons) - 1, -1, -1):
            polygon_data = self.polygons[i]
            points = polygon_data['points']
            if self.point_in_polygon(world_x, world_y, points):
                self.selected_polygon_index = i
                self.selected_polygon_indices = [i]
                break

        self.update()
    
    def select_polygon_group(self, group_id):
        """Select all polygons in a group"""
        self.selected_polygon_indices = []
        for i, polygon_data in enumerate(self.polygons):
            if 'group_id' in polygon_data and polygon_data['group_id'] == group_id:
                self.selected_polygon_indices.append(i)
    
    def point_in_polygon(self, x, y, polygon_points):
        """Check if a point is inside a polygon using ray casting algorithm"""
        if len(polygon_points) < 3:
            return False
        
        inside = False
        j = len(polygon_points) - 1
        
        for i in range(len(polygon_points)):
            xi, yi = polygon_points[i]
            xj, yj = polygon_points[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside
    
    def initialize_mandala_center(self):
        """Initialize the mandala center to the current canvas center in world coordinates"""
        if self.mandala_center_world_x is None or self.mandala_center_world_y is None:
            canvas_center_screen_x = self.width() / 2
            canvas_center_screen_y = self.height() / 2
            base_x, base_y = self.screen_to_world(canvas_center_screen_x, canvas_center_screen_y)
            self.mandala_center_world_x = base_x + self.center_offset_x
            self.mandala_center_world_y = base_y + self.center_offset_y
    
    def update_mandala_center(self):
        """Update the mandala center when offsets change"""
        if self.mandala_center_world_x is not None and self.mandala_center_world_y is not None:
            canvas_center_screen_x = self.width() / 2
            canvas_center_screen_y = self.height() / 2
            base_x, base_y = self.screen_to_world(canvas_center_screen_x, canvas_center_screen_y)
            self.mandala_center_world_x = base_x + self.center_offset_x
            self.mandala_center_world_y = base_y + self.center_offset_y
    
    def paintEvent(self, event):
        """Paint the canvas"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Initialize mandala center if not set
        self.initialize_mandala_center()
        
        # Fill canvas with white background
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        
        # Apply zoom and pan transformation
        painter.translate(self.pan_offset_x, self.pan_offset_y)
        painter.scale(self.zoom_factor, self.zoom_factor)
        
        # Draw background image if available and enabled
        if self.background_image and not self.background_image.isNull() and self.show_image:
            # Draw image at original size with offset, transformations will handle zoom/pan
            painter.drawPixmap(int(self.image_offset_x), int(self.image_offset_y), self.background_image)
        
        # Reset transformation for UI elements
        painter.resetTransform()
        
        # Draw center cross and circle like polygons - will sync perfectly with zoom/pan
        # Use the fixed mandala center world coordinates
        if self.mandala_center_world_x is not None and self.mandala_center_world_y is not None:
            mandala_center_world_x = self.mandala_center_world_x
            mandala_center_world_y = self.mandala_center_world_y
            
            # Define cross size in world coordinates (like polygon points)
            cross_size_world = 20.0  # Cross arm length in world units
            circle_radius_world = 15.0  # Circle radius in world units
        
        # Create cross lines as world coordinates
        # Horizontal line points
        h_line_points = [
            (mandala_center_world_x - cross_size_world, mandala_center_world_y),
            (mandala_center_world_x + cross_size_world, mandala_center_world_y)
        ]
        
        # Vertical line points
        v_line_points = [
            (mandala_center_world_x, mandala_center_world_y - cross_size_world),
            (mandala_center_world_x, mandala_center_world_y + cross_size_world)
        ]
        
        # Draw cross lines using world-to-screen conversion (like polygons do)
        painter.setPen(QPen(QColor(255, 0, 0), 2))  # Red color, thicker line
        
        # Draw horizontal line
        screen_x1, screen_y1 = self.world_to_screen(h_line_points[0][0], h_line_points[0][1])
        screen_x2, screen_y2 = self.world_to_screen(h_line_points[1][0], h_line_points[1][1])
        painter.drawLine(int(screen_x1), int(screen_y1), int(screen_x2), int(screen_y2))
        
        # Draw vertical line
        screen_x1, screen_y1 = self.world_to_screen(v_line_points[0][0], v_line_points[0][1])
        screen_x2, screen_y2 = self.world_to_screen(v_line_points[1][0], v_line_points[1][1])
        painter.drawLine(int(screen_x1), int(screen_y1), int(screen_x2), int(screen_y2))
        
        # Draw circle using world coordinates
        screen_center_x, screen_center_y = self.world_to_screen(mandala_center_world_x, mandala_center_world_y)
        screen_radius = abs(circle_radius_world * self.zoom_factor)  # Scale radius with zoom
        painter.setPen(QPen(QColor(255, 0, 0), 1))  # Red color
        painter.setBrush(QBrush(Qt.NoBrush))  # No fill
        painter.drawEllipse(int(screen_center_x - screen_radius), int(screen_center_y - screen_radius), 
                          int(screen_radius * 2), int(screen_radius * 2))
        
        # Draw completed polygons (convert world coordinates to screen)
        for i, polygon_data in enumerate(self.polygons):
            points = polygon_data['points']
            color = polygon_data['color']
            
            if len(points) >= 3:
                # Convert world coordinates to screen coordinates
                screen_points = []
                for world_x, world_y in points:
                    screen_x, screen_y = self.world_to_screen(world_x, world_y)
                    screen_points.append(QPoint(int(screen_x), int(screen_y)))
                
                qpolygon = QPolygon(screen_points)
                
                # Highlight selected polygons (individual or group)
                if i in self.selected_polygon_indices:
                    # Draw thicker red border for selected polygons
                    painter.setPen(QPen(QColor(255, 0, 0), 3))  # Red thick border
                else:
                    # Draw normal thin black border
                    painter.setPen(QPen(QColor(0, 0, 0), 1))  # Thin black pen for border
                
                painter.setBrush(QBrush(color))
                painter.drawPolygon(qpolygon)
        
        # Draw control points for the primary selected polygon
        if self.selected_polygon_index >= 0:
            self.draw_control_points(painter)
        
        # Draw debug circle dots
        self.draw_debug_circle_dots(painter)
        
        # Draw user-defined circle if enabled (in front of array)
        if self.show_circle and self.mandala_center_world_x is not None and self.mandala_center_world_y is not None:
            mandala_center_world_x = self.mandala_center_world_x
            mandala_center_world_y = self.mandala_center_world_y
            screen_center_x, screen_center_y = self.world_to_screen(mandala_center_world_x, mandala_center_world_y)
            user_circle_radius_world = self.circle_diameter / 2.0
            user_circle_screen_radius = abs(user_circle_radius_world * self.zoom_factor)
            painter.setBrush(QBrush(Qt.NoBrush))  # No fill for either ring

            # Outer ring (concentric with the inner one). Drawn first so the
            # inner ring + handle paint on top of it.
            outer_radius_world  = self.outer_circle_diameter / 2.0
            outer_screen_radius = abs(outer_radius_world * self.zoom_factor)
            painter.setPen(QPen(QColor(0, 150, 255), 2, Qt.DashLine))   # lighter dashed blue
            painter.drawEllipse(int(screen_center_x - outer_screen_radius),
                              int(screen_center_y - outer_screen_radius),
                              int(outer_screen_radius * 2),
                              int(outer_screen_radius * 2))

            # Inner ring (the existing one)
            painter.setPen(QPen(QColor(0, 0, 255), 2))  # Blue color, thicker line
            painter.drawEllipse(int(screen_center_x - user_circle_screen_radius),
                              int(screen_center_y - user_circle_screen_radius),
                              int(user_circle_screen_radius * 2),
                              int(user_circle_screen_radius * 2))

            # Pie-slice boundary lines (mandala mode). N lines divide the
            # annulus into N equal wedges. "Up" (-90°) always lands in the
            # MIDDLE of a wedge — so the boundary lines are offset by half a
            # wedge from up:  -90° + (k + 0.5) * (360°/N).
            # Lines only span the annulus between the inner and outer rings.
            if self.mandala_mode and self.num_copies > 0 and outer_radius_world > user_circle_radius_world:
                painter.setPen(QPen(QColor(0, 100, 200), 1, Qt.DashLine))
                cx_w = mandala_center_world_x
                cy_w = mandala_center_world_y
                inner_r_w = user_circle_radius_world
                outer_r_w = outer_radius_world
                step_deg  = 360.0 / self.num_copies
                for k in range(self.num_copies):
                    a = math.radians(-90.0 + (k + 0.5) * step_deg)
                    cosA = math.cos(a)
                    sinA = math.sin(a)
                    sx0, sy0 = self.world_to_screen(cx_w + inner_r_w * cosA, cy_w + inner_r_w * sinA)
                    sx1, sy1 = self.world_to_screen(cx_w + outer_r_w * cosA, cy_w + outer_r_w * sinA)
                    painter.drawLine(int(sx0), int(sy0), int(sx1), int(sy1))

                # Bounding square of the TOP wedge (the one centered on 12 o'clock).
                # Half-wedge angle hw = 180/N.
                # Width  W = 2 * outer * sin(hw)        (outer-arc endpoints horizontally)
                # Height H = outer - inner * cos(hw)    (from outer apex at y=-outer to inner-arc endpoint at y=-inner*cos(hw))
                # Square side S = max(W, H), centered on the wedge axis (x = cx_w)
                # vertically centered on the wedge bbox midpoint.
                hw_rad   = math.radians(180.0 / self.num_copies)
                wedge_w  = 2.0 * outer_r_w * math.sin(hw_rad)
                wedge_h  = outer_r_w - inner_r_w * math.cos(hw_rad)
                side_w   = max(wedge_w, wedge_h)
                if side_w > 0:
                    # Bounding-rect vertical center in world coords (screen-y-down).
                    bbox_top_y    = cy_w - outer_r_w
                    bbox_bot_y    = cy_w - inner_r_w * math.cos(hw_rad)
                    bbox_center_y = (bbox_top_y + bbox_bot_y) / 2.0
                    sq_left_w   = cx_w - side_w / 2.0
                    sq_right_w  = cx_w + side_w / 2.0
                    sq_top_w    = bbox_center_y - side_w / 2.0
                    sq_bottom_w = bbox_center_y + side_w / 2.0

                    sx_l, sy_t = self.world_to_screen(sq_left_w, sq_top_w)
                    sx_r, sy_b = self.world_to_screen(sq_right_w, sq_bottom_w)

                    painter.setPen(QPen(QColor(255, 0, 0), 2))      # red, solid
                    painter.setBrush(QBrush(Qt.NoBrush))
                    painter.drawRect(int(sx_l), int(sy_t),
                                     int(sx_r - sx_l), int(sy_b - sy_t))

                    # Side-length label, centered horizontally on the square,
                    # placed just above its top edge.
                    label_text = f"side = {side_w:.2f}"
                    painter.setFont(QFont("Arial", 11, QFont.Bold))
                    metrics = painter.fontMetrics()
                    text_w  = metrics.horizontalAdvance(label_text)
                    text_h  = metrics.height()
                    center_screen_x, _ = self.world_to_screen(cx_w, 0)
                    text_x  = int(center_screen_x - text_w / 2)
                    text_y  = int(sy_t - 6)   # 6 px above the square's top edge
                    # Draw a small white halo so the label stays readable on busy mosaics.
                    painter.setPen(QPen(QColor(255, 255, 255), 3))
                    painter.drawText(text_x, text_y, label_text)
                    painter.setPen(QPen(QColor(255, 0, 0)))
                    painter.drawText(text_x, text_y, label_text)
                    # Restore font for any later drawing.
                    painter.setFont(QFont())
            
            # Draw drag handle on the circle
            handle_x, handle_y = self.get_circle_drag_handle_position()
            if handle_x is not None and handle_y is not None:
                # Draw handle background (white fill with blue border)
                painter.setPen(QPen(QColor(0, 0, 255), 2))  # Blue border
                painter.setBrush(QBrush(QColor(255, 255, 255)))  # White fill
                handle_radius = self.drag_handle_size // 2
                painter.drawEllipse(int(handle_x - handle_radius), 
                                  int(handle_y - handle_radius),
                                  self.drag_handle_size, 
                                  self.drag_handle_size)
        
        # Draw drag handle on the image (bottom-left corner)
        image_handle_x, image_handle_y = self.get_image_drag_handle_position()
        if image_handle_x is not None and image_handle_y is not None:
            # Draw handle background (white fill with green border)
            painter.setPen(QPen(QColor(0, 200, 0), 2))  # Green border
            painter.setBrush(QBrush(QColor(255, 255, 255)))  # White fill
            handle_radius = self.drag_handle_size // 2
            painter.drawEllipse(int(image_handle_x - handle_radius), 
                              int(image_handle_y - handle_radius),
                              self.drag_handle_size, 
                              self.drag_handle_size)
        
        # Draw polygon cursor and current points if in polygon mode
        if self.polygon_mode:
            # Get current mouse position relative to this widget
            cursor_pos = self.mapFromGlobal(QCursor.pos())
            if self.rect().contains(cursor_pos):
                # Draw square cursor
                painter.setPen(QPen(QColor(0, 255, 0), 2))  # Green square
                painter.setBrush(QBrush(Qt.NoBrush))  # No fill
                half_size = self.polygon_cursor_size // 2
                painter.drawRect(cursor_pos.x() - half_size, 
                               cursor_pos.y() - half_size,
                               self.polygon_cursor_size, 
                               self.polygon_cursor_size)
            
            # Draw current polygon points (convert world to screen coordinates)
            if self.polygon_points:
                painter.setPen(QPen(QColor(0, 255, 0), 3))  # Green points
                painter.setBrush(QBrush(QColor(0, 255, 0)))  # Green fill
                
                # Convert world coordinates to screen coordinates for display
                screen_points = []
                for world_x, world_y in self.polygon_points:
                    screen_x, screen_y = self.world_to_screen(world_x, world_y)
                    screen_points.append((screen_x, screen_y))
                
                # Draw points
                for i, (screen_x, screen_y) in enumerate(screen_points):
                    painter.drawEllipse(int(screen_x - 3), int(screen_y - 3), 6, 6)
                    
                    # Draw point number
                    painter.setPen(QPen(QColor(255, 255, 255), 1))
                    painter.setFont(QFont('Arial', 8, QFont.Bold))
                    painter.drawText(int(screen_x + 5), int(screen_y - 5), str(i + 1))
                    painter.setPen(QPen(QColor(0, 255, 0), 3))  # Reset pen for next point
                
                # Draw lines connecting the points
                if len(screen_points) > 1:
                    painter.setPen(QPen(QColor(0, 255, 0), 2))
                    for i in range(len(screen_points) - 1):
                        x1, y1 = screen_points[i]
                        x2, y2 = screen_points[i + 1]
                        painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # In-progress line preview (blue) while the user is dragging in line mode.
        if self.line_mode and self.is_drawing_line and len(self.line_points) >= 1:
            painter.setPen(QPen(QColor(30, 120, 255), 2))
            prev = None
            for wx, wy in self.line_points:
                sx, sy = self.world_to_screen(wx, wy)
                if prev is not None:
                    painter.drawLine(int(prev[0]), int(prev[1]),
                                     int(sx), int(sy))
                prev = (sx, sy)
            # Also draw the cursor endpoint for feedback.
            if self.current_line_end is not None and prev is not None:
                ex, ey = self.world_to_screen(*self.current_line_end)
                painter.drawLine(int(prev[0]), int(prev[1]),
                                 int(ex), int(ey))

    def draw_control_points(self, painter):
        """Draw control points for the selected polygon(s)"""
        if self.selected_polygon_index < 0 or self.selected_polygon_index >= len(self.polygons):
            return
        
        # In mandala mode, draw control points for all selected polygons
        if self.mandala_mode and len(self.selected_polygon_indices) > 1:
            # Draw control points for all polygons in the group
            for idx in self.selected_polygon_indices:
                if idx < len(self.polygons):
                    polygon_data = self.polygons[idx]
                    points = polygon_data['points']
                    
                    # Primary polygon (the one we clicked on) gets yellow points
                    # Copies get red points
                    is_primary = (idx == self.selected_polygon_index)
                    
                    for i, (world_x, world_y) in enumerate(points):
                        # Convert world coordinates to screen coordinates
                        screen_x, screen_y = self.world_to_screen(world_x, world_y)
                        
                        if is_primary:
                            # Primary polygon: yellow points
                            if i == self.selected_control_point:
                                painter.setPen(QPen(QColor(255, 0, 0), 3))  # Red outline for selected
                                painter.setBrush(QBrush(QColor(255, 255, 0)))  # Yellow fill
                            else:
                                painter.setPen(QPen(QColor(0, 0, 255), 2))  # Blue outline
                                painter.setBrush(QBrush(QColor(255, 255, 0)))  # Yellow fill
                        else:
                            # Copy polygons: red points
                            painter.setPen(QPen(QColor(128, 0, 0), 2))  # Dark red outline
                            painter.setBrush(QBrush(QColor(255, 0, 0)))  # Red fill
                        
                        # Draw the control point circle
                        half_size = self.control_point_size // 2
                        painter.drawEllipse(int(screen_x - half_size), int(screen_y - half_size),
                                          self.control_point_size, self.control_point_size)
        else:
            # Single polygon mode or no group - draw yellow points as before
            polygon_data = self.polygons[self.selected_polygon_index]
            points = polygon_data['points']
            
            # Draw control points as yellow dots with blue outline
            for i, (world_x, world_y) in enumerate(points):
                # Convert world coordinates to screen coordinates
                screen_x, screen_y = self.world_to_screen(world_x, world_y)
                
                # Highlight selected control point
                if i == self.selected_control_point:
                    painter.setPen(QPen(QColor(255, 0, 0), 3))  # Red outline for selected
                    painter.setBrush(QBrush(QColor(255, 255, 0)))  # Yellow fill
                else:
                    painter.setPen(QPen(QColor(0, 0, 255), 2))  # Blue outline
                    painter.setBrush(QBrush(QColor(255, 255, 0)))  # Yellow fill
                
                # Draw the control point circle
                half_size = self.control_point_size // 2
                painter.drawEllipse(int(screen_x - half_size), int(screen_y - half_size),
                                  self.control_point_size, self.control_point_size)

    def draw_debug_circle_dots(self, painter):
        """Draw debug dots showing the circular positions"""
        if not hasattr(self, 'debug_circle_dots') or not self.debug_circle_dots:
            return
            
        # Draw bright green dots for the calculated circular positions
        painter.setPen(QPen(QColor(0, 255, 0), 3))  # Bright green outline
        painter.setBrush(QBrush(QColor(0, 255, 0)))  # Green fill
        
        for world_x, world_y in self.debug_circle_dots:
            # Convert world coordinates to screen coordinates
            screen_x, screen_y = self.world_to_screen(world_x, world_y)
            
            # Draw a larger dot so it's clearly visible
            dot_size = 10
            half_size = dot_size // 2
            painter.drawEllipse(int(screen_x - half_size), int(screen_y - half_size),
                              dot_size, dot_size)

    def _translate_selected_polygon(self, dx: float, dy: float) -> None:
        """Translate the currently-selected polygon by (dx, dy) in world
        units. If in mandala mode and the polygon has a parent_shape,
        propagate the same move to every sibling that shares that
        parent_shape — each sibling's delta is (dx, dy) rotated by
        (sibling_rotation - primary_rotation) around the origin, so the
        mandala symmetry is preserved. Called from mouseMoveEvent while
        dragging."""
        idx = self.selected_polygon_index
        if idx is None or idx < 0 or idx >= len(self.polygons):
            return
        primary = self.polygons[idx]
        primary_angle_deg = float(primary.get("rotation_angle", 0.0))
        parent = primary.get("parent_shape")
        # Move the primary.
        primary["points"] = [
            (x + dx, y + dy) for x, y in primary["points"]
        ]
        # Propagate to mandala siblings only in mandala mode + when the
        # polygon actually belongs to a group with a shared parent_shape.
        if not self.mandala_mode or parent is None:
            self.update()
            return
        for i, poly in enumerate(self.polygons):
            if i == idx:
                continue
            if poly.get("parent_shape") is not parent:
                continue
            sibling_angle_deg = float(poly.get("rotation_angle", 0.0))
            delta_rad = math.radians(sibling_angle_deg - primary_angle_deg)
            cos_a = math.cos(delta_rad)
            sin_a = math.sin(delta_rad)
            # Rotate the delta VECTOR by (θ_sibling − θ_primary). No
            # translation — just a rotation of the direction of motion.
            rot_dx = dx * cos_a - dy * sin_a
            rot_dy = dx * sin_a + dy * cos_a
            poly["points"] = [
                (x + rot_dx, y + rot_dy) for x, y in poly["points"]
            ]
        self.update()

    def _rotate_selected_polygon(self, angle_rad: float) -> None:
        """Rotate the currently-selected polygon around ITS OWN centroid
        by angle_rad. If in mandala mode and the polygon has a
        parent_shape, rotate every sibling around ITS OWN centroid by
        the same amount — preserves mandala symmetry (each copy of the
        shape rotates identically in its own local frame). Called from
        wheelEvent when the cursor is over the selected polygon."""
        idx = self.selected_polygon_index
        if idx is None or idx < 0 or idx >= len(self.polygons):
            return
        primary = self.polygons[idx]
        parent = primary.get("parent_shape")

        def rotate_around_centroid(pts, ang):
            n = len(pts)
            if n < 1:
                return pts
            cx = sum(p[0] for p in pts) / n
            cy = sum(p[1] for p in pts) / n
            ca, sa = math.cos(ang), math.sin(ang)
            return [
                (cx + (x - cx) * ca - (y - cy) * sa,
                 cy + (x - cx) * sa + (y - cy) * ca)
                for x, y in pts
            ]

        primary["points"] = rotate_around_centroid(
            primary["points"], angle_rad,
        )
        if not self.mandala_mode or parent is None:
            self.update()
            return
        for i, poly in enumerate(self.polygons):
            if i == idx:
                continue
            if poly.get("parent_shape") is not parent:
                continue
            poly["points"] = rotate_around_centroid(
                poly["points"], angle_rad,
            )
        self.update()

    def update_corresponding_points_in_copies(self, new_world_x, new_world_y):
        """Propagate a control-point drag on the SELECTED polygon to all
        its radial sibling copies. Siblings = polygons sharing the same
        `parent_shape` object reference. Each sibling's same-index
        control point is moved by rotating the dragged point around the
        mandala centre by (sibling_rotation - primary_rotation).

        Works for both:
          - polygon-mode groups (all N radial copies share ONE parent_shape)
          - line-mode groups (each base square has its own parent_shape,
            shared across its N radial copies)
        Standalone polygons (no parent_shape) have no siblings and this
        method returns without side effects."""
        import math

        if (self.selected_polygon_index < 0
                or self.selected_control_point < 0
                or self.selected_polygon_index >= len(self.polygons)):
            return

        primary = self.polygons[self.selected_polygon_index]
        parent_shape = primary.get('parent_shape')
        if parent_shape is None:
            return  # standalone polygon — no siblings to update

        cp_idx = self.selected_control_point
        primary_angle_deg = float(primary.get('rotation_angle', 0.0))

        cx = (self.mandala_center_world_x
              if self.mandala_center_world_x is not None else 0.0)
        cy = (self.mandala_center_world_y
              if self.mandala_center_world_y is not None else 0.0)

        for i, poly in enumerate(self.polygons):
            if i == self.selected_polygon_index:
                continue  # primary already updated by the drag handler
            # Sibling identity: same parent_shape by reference.
            if poly.get('parent_shape') is not parent_shape:
                continue
            pts = poly.get('points', [])
            if cp_idx >= len(pts):
                continue
            # Rotate the new position around the mandala centre by
            # (sibling_angle - primary_angle) so the sibling's same-index
            # control point lands in the analogous rotated position.
            sibling_angle_deg = float(poly.get('rotation_angle', 0.0))
            delta_rad = math.radians(sibling_angle_deg - primary_angle_deg)
            rx = new_world_x - cx
            ry = new_world_y - cy
            cos_a = math.cos(delta_rad)
            sin_a = math.sin(delta_rad)
            new_rx = rx * cos_a - ry * sin_a
            new_ry = rx * sin_a + ry * cos_a
            poly['points'][cp_idx] = (cx + new_rx, cy + new_ry)
        
        # Clear debug dots since we're now actually moving the polygons
        self.debug_circle_dots = []
        
        # Force a redraw to show the updated positions
        self.update()
    
    def get_copy_rotation_angle(self, polygon_index, group_id):
        """Get the rotation angle for a specific polygon copy"""
        # Find which copy number this polygon is
        group_polygons = []
        for i, poly in enumerate(self.polygons):
            if 'group_id' in poly and poly['group_id'] == group_id:
                group_polygons.append(i)
        
        if polygon_index not in group_polygons:
            return None
            
        copy_number = group_polygons.index(polygon_index)
        
        # Get the number of copies from the group info
        group_info = self.get_polygon_group_by_id(group_id)
        if not group_info:
            return None
            
        num_copies = group_info['parent_shape']['num_copies']
        angle_step = 2 * 3.14159 / num_copies  # 2π / num_copies
        
        return copy_number * angle_step
    
    def find_control_point_at_screen_pos(self, screen_x, screen_y):
        """Find which control point is at the given screen position"""
        if self.selected_polygon_index < 0 or self.selected_polygon_index >= len(self.polygons):
            return -1
            
        polygon_data = self.polygons[self.selected_polygon_index]
        points = polygon_data['points']
        
        for i, (world_x, world_y) in enumerate(points):
            # Convert world coordinates to screen coordinates
            point_screen_x, point_screen_y = self.world_to_screen(world_x, world_y)
            
            # Check if click is within control point circle
            distance = ((screen_x - point_screen_x)**2 + (screen_y - point_screen_y)**2)**0.5
            if distance <= self.control_point_size:
                return i
                
        return -1


# ───────────────────────────────────────────────────────────────────────────
# Color-picker helper widgets — copied from duplicator.py so the left panel
# here can offer the same color tooling (HSV picker, saved palette, eyedropper,
# replace-color, color variance, background color).
# ───────────────────────────────────────────────────────────────────────────

class _HueBar(QWidget):
    """Thin horizontal rainbow strip for selecting hue."""
    hue_changed  = pyqtSignal(float)
    hover_color  = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hue = 0.0
        self.setFixedHeight(16)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)

    def set_hue(self, hue):
        self._hue = max(0.0, min(359.0, float(hue)))
        self.update()

    def _hue_from_x(self, x):
        return max(0.0, min(359.0, x / max(1, self.width()) * 359.0))

    def paintEvent(self, event):
        p = QPainter(self)
        w, h = self.width(), self.height()
        grad = QLinearGradient(0, 0, w, 0)
        for i in range(7):
            grad.setColorAt(i / 6.0, QColor.fromHsv(min(359, i * 60), 255, 255))
        p.fillRect(0, 0, w, h, grad)
        x = int(self._hue / 359.0 * (w - 1))
        p.setPen(QPen(QColor(0, 0, 0), 2)); p.drawLine(x, 0, x, h - 1)
        p.setPen(QPen(QColor(255, 255, 255), 1)); p.drawLine(max(0, x - 1), 0, max(0, x - 1), h - 1)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._hue = self._hue_from_x(event.x()); self.hue_changed.emit(self._hue); self.update()

    def mouseMoveEvent(self, event):
        hue = self._hue_from_x(event.x())
        self.hover_color.emit(QColor.fromHsv(int(hue), 255, 255))
        if event.buttons() & Qt.LeftButton:
            self._hue = hue; self.hue_changed.emit(self._hue); self.update()

    def leaveEvent(self, event):
        self.hover_color.emit(None)


class _SVSquare(QWidget):
    """Saturation / Value gradient square for the current hue."""
    sv_changed  = pyqtSignal(float, float)
    hover_color = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hue = 0; self._sat = 1.0; self._val = 1.0
        self.setFixedHeight(130)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.CrossCursor); self.setMouseTracking(True)

    def set_hue(self, hue):
        self._hue = int(max(0, min(359, hue))); self.update()

    def set_sv(self, sat, val):
        self._sat = float(sat); self._val = float(val); self.update()

    def _sv_from_pos(self, x, y):
        w, h = max(1, self.width()), max(1, self.height())
        return max(0.0, min(1.0, x / w)), max(0.0, min(1.0, 1.0 - y / h))

    def paintEvent(self, event):
        p = QPainter(self); w, h = self.width(), self.height()
        hg = QLinearGradient(0, 0, w, 0)
        hg.setColorAt(0.0, QColor(255, 255, 255))
        hg.setColorAt(1.0, QColor.fromHsv(self._hue, 255, 255))
        p.fillRect(0, 0, w, h, hg)
        vg = QLinearGradient(0, 0, 0, h)
        vg.setColorAt(0.0, QColor(0, 0, 0, 0)); vg.setColorAt(1.0, QColor(0, 0, 0, 255))
        p.fillRect(0, 0, w, h, vg)
        cx = int(self._sat * (w - 1)); cy = int((1.0 - self._val) * (h - 1))
        p.setPen(QPen(QColor(0, 0, 0), 1)); p.drawEllipse(cx - 5, cy - 5, 10, 10)
        p.setPen(QPen(QColor(255, 255, 255), 1)); p.drawEllipse(cx - 4, cy - 4, 8, 8)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._sat, self._val = self._sv_from_pos(event.x(), event.y())
            self.sv_changed.emit(self._sat, self._val); self.update()

    def mouseMoveEvent(self, event):
        sat, val = self._sv_from_pos(event.x(), event.y())
        self.hover_color.emit(QColor.fromHsv(self._hue, int(sat * 255), int(val * 255)))
        if event.buttons() & Qt.LeftButton:
            self._sat, self._val = sat, val
            self.sv_changed.emit(self._sat, self._val); self.update()

    def leaveEvent(self, event):
        self.hover_color.emit(None)


class ColorPickerWidget(QWidget):
    """HSV color picker: hue bar + SV square + RGB spin boxes."""
    color_changed = pyqtSignal(QColor)

    def __init__(self, parent=None):
        super().__init__(parent); self._updating = False
        self._build_ui(); self._apply_color(QColor(0, 0, 0))

    def _build_ui(self):
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(4)
        self._hue_bar = _HueBar(); self._hue_bar.hue_changed.connect(self._hue_changed); layout.addWidget(self._hue_bar)
        self._sv = _SVSquare(); self._sv.sv_changed.connect(self._sv_changed); layout.addWidget(self._sv)
        rgb_row = QHBoxLayout(); rgb_row.setSpacing(4); self._spins = []
        for label in ('R', 'G', 'B'):
            col = QVBoxLayout(); col.setSpacing(1)
            lbl = QLabel(label); lbl.setAlignment(Qt.AlignCenter)
            sp = QSpinBox(); sp.setRange(0, 255); sp.setAlignment(Qt.AlignCenter); sp.setMinimumWidth(52)
            col.addWidget(lbl); col.addWidget(sp); rgb_row.addLayout(col); self._spins.append(sp)
        layout.addLayout(rgb_row)
        for sp in self._spins:
            sp.valueChanged.connect(self._rgb_changed)

    def get_color(self):
        return QColor(self._spins[0].value(), self._spins[1].value(), self._spins[2].value())

    def _apply_color(self, color):
        self._updating = True
        h, s, v, _ = color.getHsv()
        if h >= 0:
            self._hue_bar.set_hue(h); self._sv.set_hue(h)
        self._sv.set_sv(s / 255.0, v / 255.0)
        self._spins[0].setValue(color.red()); self._spins[1].setValue(color.green()); self._spins[2].setValue(color.blue())
        self._updating = False

    def _hue_changed(self, hue):
        if self._updating: return
        self._sv.set_hue(int(hue))
        color = QColor.fromHsv(int(hue), int(self._sv._sat * 255), int(self._sv._val * 255))
        self._updating = True
        self._spins[0].setValue(color.red()); self._spins[1].setValue(color.green()); self._spins[2].setValue(color.blue())
        self._updating = False
        self.color_changed.emit(color)

    def _sv_changed(self, sat, val):
        if self._updating: return
        color = QColor.fromHsv(int(self._hue_bar._hue), int(sat * 255), int(val * 255))
        self._updating = True
        self._spins[0].setValue(color.red()); self._spins[1].setValue(color.green()); self._spins[2].setValue(color.blue())
        self._updating = False
        self.color_changed.emit(color)

    def _rgb_changed(self):
        if self._updating: return
        color = QColor(self._spins[0].value(), self._spins[1].value(), self._spins[2].value())
        self._apply_color(color); self.color_changed.emit(color)


class _ClickableColorBox(QLabel):
    right_clicked = pyqtSignal()
    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.right_clicked.emit()
        else:
            super().mousePressEvent(event)


class _SavedPaletteWidget(QWidget):
    """A 16-slot saved-color strip beneath the picker."""
    slot_loaded  = pyqtSignal(QColor)
    hover_color  = pyqtSignal(object)

    SLOTS = 16; COLS = 8; SWATCH = 22; GAP = 2

    def __init__(self, get_current_color_fn, parent=None):
        super().__init__(parent)
        self._get_color = get_current_color_fn
        self._slots = [None] * self.SLOTS
        self._hover_idx = -1; self._armed_idx = -1
        rows = math.ceil(self.SLOTS / self.COLS)
        w = self.COLS * (self.SWATCH + self.GAP) + self.GAP
        h = rows * (self.SWATCH + self.GAP) + self.GAP
        self.setFixedSize(w, h); self.setCursor(Qt.PointingHandCursor); self.setMouseTracking(True)
        self.setToolTip(
            'Left-click filled slot: load color\n'
            'Right-click any slot: arm it, then pick a color in the picker to fill it',
        )

    def _idx_from_pos(self, x, y):
        col = (x - self.GAP) // (self.SWATCH + self.GAP)
        row = (y - self.GAP) // (self.SWATCH + self.GAP)
        idx = row * self.COLS + col
        if 0 <= col < self.COLS and 0 <= idx < self.SLOTS:
            return idx
        return -1

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing, False)
        p.fillRect(self.rect(), QColor(180, 180, 180))
        for i in range(self.SLOTS):
            row = i // self.COLS; col = i % self.COLS
            x = self.GAP + col * (self.SWATCH + self.GAP)
            y = self.GAP + row * (self.SWATCH + self.GAP)
            color = self._slots[i]
            if color is not None:
                p.fillRect(x, y, self.SWATCH, self.SWATCH, color)
                p.setPen(QPen(QColor(60, 60, 60), 1))
            else:
                p.fillRect(x, y, self.SWATCH, self.SWATCH, QColor(220, 220, 220))
                p.setPen(QPen(QColor(160, 160, 160), 1))
                mid = self.SWATCH // 2
                p.drawLine(x + mid, y + 2, x + mid, y + self.SWATCH - 3)
                p.drawLine(x + 2, y + mid, x + self.SWATCH - 3, y + mid)
                p.setPen(QPen(QColor(160, 160, 160), 1))
            if i == self._armed_idx:
                p.setPen(QPen(QColor(255, 140, 0), 2))
                p.drawRect(x + 1, y + 1, self.SWATCH - 3, self.SWATCH - 3)
            elif i == self._hover_idx:
                p.setPen(QPen(QColor(255, 255, 255), 2))
                p.drawRect(x + 1, y + 1, self.SWATCH - 3, self.SWATCH - 3)
            p.setPen(QPen(QColor(60, 60, 60), 1))
            p.drawRect(x, y, self.SWATCH - 1, self.SWATCH - 1)

    def mouseMoveEvent(self, event):
        idx = self._idx_from_pos(event.x(), event.y())
        if idx != self._hover_idx:
            self._hover_idx = idx; self.update()
        color = self._slots[idx] if 0 <= idx < self.SLOTS else None
        self.hover_color.emit(color)

    def leaveEvent(self, event):
        self._hover_idx = -1; self.update(); self.hover_color.emit(None)

    def fill_armed_slot(self, color):
        if self._armed_idx < 0:
            return
        self._slots[self._armed_idx] = QColor(color); self._armed_idx = -1; self.update()

    def mousePressEvent(self, event):
        idx = self._idx_from_pos(event.x(), event.y())
        if idx < 0:
            return
        if event.button() == Qt.LeftButton:
            if self._slots[idx] is not None:
                self.slot_loaded.emit(QColor(self._slots[idx])); self.update()
        elif event.button() == Qt.RightButton:
            self._armed_idx = idx; self.update()


class SidePanel(QFrame):
    """Side panel widget"""
    
    def __init__(self, title, canvas=None):
        super().__init__()
        self.setFrameStyle(QFrame.StyledPanel)
        self.setMinimumWidth(200)
        self.setMaximumWidth(260)
        self.setStyleSheet("background-color: #f0f0f0;")
        self.canvas = canvas

        # Create layout for the panel
        layout = QVBoxLayout()

        # Left panel = color tooling (HSV picker, saved palette, eyedropper,
        # replace, color-variance, background color) — same as duplicator.py.
        if title == "Left Panel" and canvas:
            layout.addWidget(QLabel('Color Palette:'))
            self.selected_color = QColor(0, 0, 0)
            self.create_color_palette(layout)
            layout.addStretch()

        # Add buttons for right panel
        if title == "Right Panel" and canvas:
            load_bg_button = QPushButton("Load Background")
            load_bg_button.clicked.connect(self.load_background)
            layout.addWidget(load_bg_button)

            # Background Scale (%): percentage is ALWAYS relative to the
            # image's original loaded size (not incremental). Entering
            # 100 restores; 50 always halves the original; 200 always
            # doubles it. Handler re-scales from a stored snapshot on
            # every value change so values don't compound.
            layout.addWidget(QLabel("Background Scale (%):"))
            self.bg_scale_input = QLineEdit()
            self.bg_scale_input.setText("100")
            self.bg_scale_input.setPlaceholderText(
                "% of original (100 = as loaded)"
            )
            self.bg_scale_input.setToolTip(
                "Scale the background image to this percentage of its "
                "ORIGINAL loaded size. Always absolute — 100 restores, "
                "50 halves the original, 200 doubles it. Not incremental."
            )
            self.bg_scale_input.textChanged.connect(
                self.on_bg_scale_changed,
            )
            layout.addWidget(self.bg_scale_input)

            # Undo the last polygon-list mutation (draw, erase, repaint,
            # line-draw). Also bound to Ctrl+Z globally (see main()).
            self.undo_button = QPushButton("Undo (Ctrl+Z)")
            self.undo_button.setToolTip(
                "Undo the last polygon action (add / draw line / erase / "
                "repaint). Ctrl+Z has the same effect."
            )
            self.undo_button.clicked.connect(self.on_undo_clicked)
            layout.addWidget(self.undo_button)

            # Add polygon checkbox
            self.polygon_checkbox = QCheckBox("Polygon")
            self.polygon_checkbox.toggled.connect(self.on_polygon_toggled)
            layout.addWidget(self.polygon_checkbox)

            # Line-drawing tool — click + drag on the canvas to trace a
            # path; on release, rotated squares of the size below are
            # placed along it. If Mandala is on, each square gets
            # radial copies too.
            self.line_checkbox = QCheckBox("Line")
            self.line_checkbox.setToolTip(
                "Draw a path with click+drag; on release, squares of the "
                "chosen size are dropped along it. With Mandala checked, "
                "radial copies of every square are also created."
            )
            self.line_checkbox.toggled.connect(self.on_line_toggled)
            layout.addWidget(self.line_checkbox)

            # Line-polygon size (world-px side of each square)
            layout.addWidget(QLabel("Polygon Size (px):"))
            self.line_polygon_size_input = QLineEdit()
            self.line_polygon_size_input.setText(str(canvas.line_polygon_size))
            self.line_polygon_size_input.setPlaceholderText("15")
            self.line_polygon_size_input.textChanged.connect(
                self.on_line_polygon_size_changed,
            )
            layout.addWidget(self.line_polygon_size_input)

            # Gap between consecutive line polygons
            layout.addWidget(QLabel("Polygon Gap (px):"))
            self.line_polygon_gap_input = QLineEdit()
            self.line_polygon_gap_input.setText(str(canvas.line_polygon_gap))
            self.line_polygon_gap_input.setPlaceholderText("2")
            self.line_polygon_gap_input.textChanged.connect(
                self.on_line_polygon_gap_changed,
            )
            layout.addWidget(self.line_polygon_gap_input)

            # Add mandala checkbox
            self.mandala_checkbox = QCheckBox("Mandala")
            self.mandala_checkbox.setChecked(True)  # Checked by default
            self.mandala_checkbox.toggled.connect(self.on_mandala_toggled)
            layout.addWidget(self.mandala_checkbox)
            
            # Add eraser mode checkbox
            self.eraser_checkbox = QCheckBox("Eraser Mode")
            self.eraser_checkbox.toggled.connect(self.on_eraser_toggled)
            layout.addWidget(self.eraser_checkbox)

            # Paint mode: when checked, left-click on any polygon fills
            # it with the palette colour. In mandala mode the entire
            # group is repainted so radial copies stay in sync with the
            # clicked polygon. Same behaviour as the existing right-click
            # paint, just triggered by left-click instead.
            self.paint_checkbox = QCheckBox("Paint")
            self.paint_checkbox.setToolTip(
                "Left-click any polygon to fill it with the currently-"
                "selected palette colour (Left Panel). With Mandala "
                "checked, all radial copies of the clicked polygon are "
                "repainted too. Untick to return to normal click "
                "behaviour (selection / drag)."
            )
            self.paint_checkbox.toggled.connect(self.on_paint_toggled)
            layout.addWidget(self.paint_checkbox)
            
            # Add circle checkbox and diameter input
            self.circle_checkbox = QCheckBox("Circle")
            self.circle_checkbox.toggled.connect(self.on_circle_toggled)
            layout.addWidget(self.circle_checkbox)
            
            # Circle diameter input (inner)
            circle_label = QLabel("Diameter:")
            layout.addWidget(circle_label)
            self.circle_diameter_input = QLineEdit("1000")
            self.circle_diameter_input.textChanged.connect(self.on_circle_diameter_changed)
            layout.addWidget(self.circle_diameter_input)

            # Outer-circle diameter input (concentric with the inner circle)
            outer_label = QLabel("Outer diameter:")
            layout.addWidget(outer_label)
            self.outer_circle_diameter_input = QLineEdit("1500")
            self.outer_circle_diameter_input.textChanged.connect(self.on_outer_circle_diameter_changed)
            layout.addWidget(self.outer_circle_diameter_input)

            # ── Move circles: 4-arrow grid + step-size input ─────────────
            # Nudges the mandala circles (inner + outer share the same
            # centre) by ±step in world units per click. Centre button
            # resets the offset so the circles return to the canvas centre.
            layout.addWidget(QLabel("Move circles:"))
            arrows_grid = QGridLayout()
            arrows_grid.setSpacing(2)

            up_btn = QPushButton("↑")     # ↑
            up_btn.setFixedSize(36, 28)
            up_btn.setToolTip("Move circles up by the step below.")
            up_btn.clicked.connect(lambda: self._on_circle_arrow(0, -1))
            arrows_grid.addWidget(up_btn, 0, 1)

            left_btn = QPushButton("←")   # ←
            left_btn.setFixedSize(36, 28)
            left_btn.setToolTip("Move circles left by the step below.")
            left_btn.clicked.connect(lambda: self._on_circle_arrow(-1, 0))
            arrows_grid.addWidget(left_btn, 1, 0)

            reset_btn = QPushButton("○")  # ○
            reset_btn.setFixedSize(36, 28)
            reset_btn.setToolTip(
                "Reset the circle centre offset to the canvas centre."
            )
            reset_btn.clicked.connect(self._on_circle_reset)
            arrows_grid.addWidget(reset_btn, 1, 1)

            right_btn = QPushButton("→")  # →
            right_btn.setFixedSize(36, 28)
            right_btn.setToolTip("Move circles right by the step below.")
            right_btn.clicked.connect(lambda: self._on_circle_arrow(1, 0))
            arrows_grid.addWidget(right_btn, 1, 2)

            down_btn = QPushButton("↓")   # ↓
            down_btn.setFixedSize(36, 28)
            down_btn.setToolTip("Move circles down by the step below.")
            down_btn.clicked.connect(lambda: self._on_circle_arrow(0, 1))
            arrows_grid.addWidget(down_btn, 2, 1)

            layout.addLayout(arrows_grid)

            # Step size input — pixels moved per arrow click.
            step_row = QHBoxLayout()
            step_row.addWidget(QLabel("Step (px):"))
            self.circle_step_input = QLineEdit()
            self.circle_step_input.setText("10")
            self.circle_step_input.setPlaceholderText("px per click")
            self.circle_step_input.setToolTip(
                "Pixels the circles move per arrow click (world units)."
            )
            self.circle_step_input.setFixedWidth(60)
            step_row.addWidget(self.circle_step_input)
            step_row.addStretch(1)
            layout.addLayout(step_row)

            # Detect polygons in the ring between the inner + outer circles
            # using mosaic_to_csv.py's adaptive-threshold + connected-
            # components + Douglas-Peucker pipeline. Results are added to
            # the canvas as standalone polygons filled with each tile's
            # mean colour.
            self.detect_polys_btn = QPushButton("Detect Polygons in Ring")
            self.detect_polys_btn.setToolTip(
                "Detect tesserae in the ring between the inner and outer "
                "circles from the background image, using the same "
                "algorithm as mosaic_to_csv.py (adaptive Gaussian "
                "threshold + connected components + Douglas-Peucker "
                "simplification). Each detected polygon is added to the "
                "canvas filled with its mean colour."
            )
            self.detect_polys_btn.clicked.connect(
                self.on_detect_polygons_between_circles,
            )
            layout.addWidget(self.detect_polys_btn)
            
            # Add show image checkbox
            self.show_image_checkbox = QCheckBox("Show Image")
            self.show_image_checkbox.setChecked(True)  # Checked by default
            self.show_image_checkbox.toggled.connect(self.on_show_image_toggled)
            layout.addWidget(self.show_image_checkbox)
            
            # Add save and load array buttons
            save_array_button = QPushButton("Save Array")
            save_array_button.clicked.connect(self.save_array)
            layout.addWidget(save_array_button)

            load_array_button = QPushButton("Load Array")
            load_array_button.clicked.connect(self.load_array)
            layout.addWidget(load_array_button)

            # Project save/load — captures the entire canvas state (background,
            # polygons + groups, mandala settings, circles, view) into one .mmp file.
            save_project_button = QPushButton("Save Project")
            save_project_button.clicked.connect(self.save_project)
            layout.addWidget(save_project_button)

            load_project_button = QPushButton("Load Project")
            load_project_button.clicked.connect(self.load_project)
            layout.addWidget(load_project_button)
            
            # Add number of copies control
            copies_label = QLabel("Radial Copies:")
            copies_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
            layout.addWidget(copies_label)
            
            self.copies_input = QLineEdit()
            self.copies_input.setText("6")  # Default value
            self.copies_input.setPlaceholderText("Enter number (1-36)")
            self.copies_input.textChanged.connect(self.on_copies_changed)
            layout.addWidget(self.copies_input)
        
        # Add stretch to push content to top
        layout.addStretch()
        
        self.setLayout(layout)
    
    def load_background(self):
        """Load background image for canvas"""
        if not self.canvas:
            return
            
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Background Image",
            "",
            "Image files (*.png *.jpg *.jpeg *.bmp *.tiff *.gif);;All files (*.*)"
        )
        
        if file_path:
            # Ask for desired size of the longer side
            size, ok = QInputDialog.getInt(
                self,
                "Image Size",
                "Enter the desired length for the longer side (pixels):",
                value=1000,  # Default value
                min=100,     # Minimum value
                max=10000,   # Maximum value
                step=100     # Step size
            )
            
            if ok:
                self.canvas.set_background_image(file_path, desired_size=size)
            else:
                # User cancelled size dialog, load with original size
                self.canvas.set_background_image(file_path)
            # New original = new 100 % reference. Reset the spinbox to
            # 100 without firing textChanged, so a stale value from the
            # previous image doesn't immediately downscale/upscale the
            # newly-loaded one.
            if hasattr(self, "bg_scale_input"):
                self.bg_scale_input.blockSignals(True)
                self.bg_scale_input.setText("100")
                self.bg_scale_input.blockSignals(False)

    def on_bg_scale_changed(self, text: str):
        """Sidebar Background Scale (%) handler — parse the input, clamp
        to a sensible range, and re-scale the background from the stored
        original snapshot. Ignores mid-typing invalid text so the user
        can freely edit before committing."""
        if not self.canvas:
            return
        text = (text or "").strip()
        if not text:
            return
        try:
            pct = float(text)
        except (ValueError, TypeError):
            return
        # Clamp to [1, 2000] — stops a stray "0" or huge value from
        # blowing up memory / crashing.
        pct = max(1.0, min(2000.0, pct))
        self.canvas.set_background_scale(pct)

    def _current_circle_step(self) -> float:
        """Read the sidebar's Step (px) field and clamp to a sensible
        range. Falls back to 10 px for empty / invalid text."""
        text = (
            self.circle_step_input.text().strip()
            if hasattr(self, "circle_step_input") else "10"
        )
        try:
            step = float(text) if text else 10.0
        except (ValueError, TypeError):
            step = 10.0
        return max(0.1, min(10000.0, step))

    def _on_circle_arrow(self, dx: int, dy: int) -> None:
        """Arrow button handler — move the circles by (dx, dy) × step
        world units. dx/dy are direction integers (-1, 0, +1)."""
        if not self.canvas:
            return
        step = self._current_circle_step()
        self.canvas.move_circles(dx * step, dy * step)

    def _on_circle_reset(self) -> None:
        """Centre-of-arrows button — reset the circle offset to (0, 0)."""
        if self.canvas:
            self.canvas.reset_circle_offset()

    def on_undo_clicked(self):
        """Undo button handler — pops the top of the canvas undo stack."""
        if self.canvas:
            self.canvas.undo_last()

    def on_polygon_toggled(self, checked):
        """Handle polygon checkbox toggle"""
        if self.canvas:
            self.canvas.toggle_polygon_mode()
            # Turning Polygon on turns Line off — keep the checkbox in sync.
            if checked and hasattr(self, 'line_checkbox') \
                    and self.line_checkbox.isChecked():
                self.line_checkbox.blockSignals(True)
                self.line_checkbox.setChecked(False)
                self.line_checkbox.blockSignals(False)

    def on_line_toggled(self, checked):
        """Handle Line checkbox toggle — mutex with Polygon."""
        if not self.canvas:
            return
        self.canvas.set_line_mode(checked)
        if checked and self.polygon_checkbox.isChecked():
            # set_line_mode already turned polygon_mode off on the canvas;
            # sync the checkbox visual state without re-firing the handler.
            self.polygon_checkbox.blockSignals(True)
            self.polygon_checkbox.setChecked(False)
            self.polygon_checkbox.blockSignals(False)

    def on_line_polygon_size_changed(self, text):
        """Update the line-polygon square size (world-px)."""
        if not self.canvas:
            return
        try:
            self.canvas.set_line_polygon_size(int(float(text)))
        except (ValueError, TypeError):
            pass  # ignore mid-typing invalid text

    def on_line_polygon_gap_changed(self, text):
        """Update the gap between consecutive line polygons (world-px)."""
        if not self.canvas:
            return
        try:
            self.canvas.set_line_polygon_gap(int(float(text)))
        except (ValueError, TypeError):
            pass

    def on_mandala_toggled(self, checked):
        """Handle mandala checkbox toggle"""
        if self.canvas:
            self.canvas.set_mandala_mode(checked)
    
    def on_paint_toggled(self, checked):
        """Paint checkbox handler — flips canvas.paint_mode. When ON,
        left-click on any polygon fills it (and its mandala copies)
        with the palette colour."""
        if self.canvas:
            self.canvas.paint_mode = bool(checked)

    def on_eraser_toggled(self, checked):
        """Handle eraser mode checkbox toggle"""
        if self.canvas:
            self.canvas.set_eraser_mode(checked)
    
    def on_circle_toggled(self, checked):
        """Handle circle checkbox toggle"""
        if self.canvas:
            self.canvas.set_circle_visible(checked)
    
    def on_outer_circle_diameter_changed(self, text):
        """Handle outer-circle diameter input change."""
        if self.canvas:
            try:
                diameter = float(text) if text else 1500
                self.canvas.set_outer_circle_diameter(diameter)
            except ValueError:
                pass   # ignore mid-typing invalid text

    def on_detect_polygons_between_circles(self):
        """Detect-Polygons-in-Ring button handler — delegates to the canvas
        method which does the actual detection + polygon insertion."""
        if not self.canvas:
            return
        try:
            n = self.canvas.detect_polygons_between_circles()
        except Exception as e:
            QMessageBox.critical(
                self, "Detection failed", f"{type(e).__name__}: {e}",
            )
            return
        QMessageBox.information(
            self, "Detection complete",
            f"Detected {n} polygon(s) in the ring between the inner and "
            f"outer circles. They are now on the canvas.",
        )

    def on_circle_diameter_changed(self, text):
        """Handle circle diameter input change"""
        if self.canvas:
            try:
                diameter = float(text) if text else 1000
                self.canvas.set_circle_diameter(diameter)
            except ValueError:
                # Invalid number, ignore or use default
                pass
    
    def on_show_image_toggled(self, checked):
        """Handle show image checkbox toggle"""
        if self.canvas:
            self.canvas.set_image_visible(checked)
    
    def on_copies_changed(self):
        """Handle copies input text changes"""
        if self.canvas:
            try:
                # Parse the input and validate range
                num_copies = int(self.copies_input.text())
                num_copies = max(1, min(36, num_copies))  # Clamp between 1 and 36
                self.canvas.set_num_copies(num_copies)
                
                # Update the input field if we clamped the value
                if str(num_copies) != self.copies_input.text():
                    self.copies_input.setText(str(num_copies))
                    
            except ValueError:
                # Invalid input, reset to default
                self.canvas.set_num_copies(6)
                if self.copies_input.text() == "":  # Don't reset while user is typing
                    self.copies_input.setText("6")
    
    # ── Color palette UI (ported from duplicator.py) ──────────────────────

    def create_color_palette(self, layout):
        """HSV picker + chosen / hover color preview + 16-slot saved palette +
        replace-color row + color-variance row + background color picker."""
        self.color_picker = ColorPickerWidget()
        self.color_picker.color_changed.connect(self._on_picker_color_changed)
        layout.addWidget(self.color_picker)

        boxes_row = QHBoxLayout(); boxes_row.setSpacing(6)
        boxes_row.addWidget(QLabel('Color:'))
        self._chosen_color_box = _ClickableColorBox()
        self._chosen_color_box.setFixedSize(36, 22)
        self._chosen_color_box.setStyleSheet('background-color: rgb(0,0,0); border: 2px solid #555;')
        self._chosen_color_box.setToolTip('Active color — right-click to pick color from canvas')
        self._chosen_color_box.right_clicked.connect(self._start_eyedropper)
        boxes_row.addWidget(self._chosen_color_box)

        self._eyedropper_btn = QPushButton('⊕')
        self._eyedropper_btn.setFixedSize(22, 22)
        self._eyedropper_btn.setToolTip('Pick color from canvas (then right-click a polygon)')
        self._eyedropper_btn.setCheckable(True)
        self._eyedropper_btn.clicked.connect(self._on_eyedropper_btn_clicked)
        boxes_row.addWidget(self._eyedropper_btn)

        boxes_row.addSpacing(4)
        boxes_row.addWidget(QLabel('Hover:'))
        self._hover_preview = QLabel()
        self._hover_preview.setFixedSize(36, 22)
        self._hover_preview.setStyleSheet('background-color: transparent; border: 1px solid #888;')
        boxes_row.addWidget(self._hover_preview)
        boxes_row.addStretch()
        layout.addLayout(boxes_row)

        # Saved-color strip
        self._saved_palette = _SavedPaletteWidget(self.color_picker.get_color)
        self._saved_palette.slot_loaded.connect(self._load_saved_color)
        self._saved_palette.hover_color.connect(self._on_palette_hover)
        self.color_picker._hue_bar.hover_color.connect(self._on_palette_hover)
        self.color_picker._sv.hover_color.connect(self._on_palette_hover)
        self.color_picker.color_changed.connect(self._saved_palette.fill_armed_slot)
        layout.addWidget(self._saved_palette)

        # Replace-color row
        self._replace_from_color = None
        replace_row = QHBoxLayout(); replace_row.setSpacing(4)
        replace_row.addWidget(QLabel('Replace:'))
        self._replace_from_box = QLabel()
        self._replace_from_box.setFixedSize(36, 22)
        self._replace_from_box.setStyleSheet('background-color: transparent; border: 2px dashed #888;')
        self._replace_from_box.setToolTip('Source color to replace — use ⊕ to sample from canvas')
        replace_row.addWidget(self._replace_from_box)
        self._replace_sample_btn = QPushButton('⊕')
        self._replace_sample_btn.setFixedSize(22, 22)
        self._replace_sample_btn.setToolTip('Click then click a polygon to set source color')
        self._replace_sample_btn.setCheckable(True)
        self._replace_sample_btn.clicked.connect(self._on_replace_sample_btn_clicked)
        replace_row.addWidget(self._replace_sample_btn)
        replace_row.addWidget(QLabel('→ current'))
        self._replace_apply_btn = QPushButton('Apply')
        self._replace_apply_btn.setFixedSize(40, 22)
        self._replace_apply_btn.setToolTip('Replace all polygons with source color → current color')
        self._replace_apply_btn.clicked.connect(self._apply_replace_color)
        replace_row.addWidget(self._replace_apply_btn)
        replace_row.addStretch()
        layout.addLayout(replace_row)

        # Color variance row
        variance_row = QHBoxLayout(); variance_row.setSpacing(4)
        variance_btn = QPushButton('Vary Colors')
        variance_btn.setToolTip('Randomly shift every polygon color by up to the variance amount')
        variance_btn.clicked.connect(self._apply_color_variance)
        variance_row.addWidget(variance_btn)
        self._variance_slider = QSlider(Qt.Horizontal)
        self._variance_slider.setRange(0, 128); self._variance_slider.setValue(20)
        self._variance_slider.setFixedWidth(80)
        self._variance_slider.setToolTip('Max color variance per channel (0–128)')
        variance_row.addWidget(self._variance_slider)
        self._variance_label = QLabel('20'); self._variance_label.setFixedWidth(24)
        self._variance_slider.valueChanged.connect(lambda v: self._variance_label.setText(str(v)))
        variance_row.addWidget(self._variance_label)
        variance_row.addStretch()
        layout.addLayout(variance_row)

        # Background color row
        bg_row = QHBoxLayout(); bg_row.setSpacing(4)
        bg_row.addWidget(QLabel('Background:'))
        self._bg_color_box = QLabel(); self._bg_color_box.setFixedSize(36, 22)
        self._bg_color_box.setStyleSheet('background-color: rgb(255,255,255); border: 2px solid #555;')
        bg_row.addWidget(self._bg_color_box)
        bg_btn = QPushButton('Change'); bg_btn.setFixedSize(55, 22)
        bg_btn.setToolTip('Change the canvas background color')
        bg_btn.clicked.connect(self._change_background_color)
        bg_row.addWidget(bg_btn); bg_row.addStretch()
        layout.addLayout(bg_row)

    def _change_background_color(self):
        if not self.canvas:
            return
        current = getattr(self.canvas, 'background_color', QColor(255, 255, 255))
        color = QColorDialog.getColor(current, self, 'Choose Background Color')
        if color.isValid():
            self.canvas.background_color = color
            r, g, b = color.red(), color.green(), color.blue()
            self._bg_color_box.setStyleSheet(f'background-color: rgb({r},{g},{b}); border: 2px solid #555;')
            # Update canvas widget background.
            self.canvas.setStyleSheet(
                f"background-color: rgb({r},{g},{b}); border: 1px solid black;",
            )
            self.canvas.update()

    def _on_replace_sample_btn_clicked(self, checked):
        if checked:
            if self.canvas:
                self.canvas.replace_eyedropper_mode = True
                self.canvas.setCursor(Qt.CrossCursor)
            self._replace_sample_btn.setStyleSheet('background-color: #f90; border: 2px solid #c60;')
        else:
            if self.canvas:
                self.canvas.replace_eyedropper_mode = False
                self.canvas.setCursor(Qt.ArrowCursor)
            self._replace_sample_btn.setStyleSheet('')

    def receive_replace_source_color(self, color):
        self._replace_from_color = color
        r, g, b = color.red(), color.green(), color.blue()
        self._replace_from_box.setStyleSheet(
            f'background-color: rgb({r},{g},{b}); border: 2px solid #555;',
        )
        self._replace_sample_btn.setChecked(False)
        self._replace_sample_btn.setStyleSheet('')

    def _apply_replace_color(self):
        if self._replace_from_color is None or not self.canvas:
            return
        src = self._replace_from_color
        dst = self.selected_color
        count = 0
        for polygon in self.canvas.polygons:
            c = polygon.get('color')
            if c is not None and c.red() == src.red() and c.green() == src.green() and c.blue() == src.blue():
                polygon['color'] = QColor(dst)
                count += 1
        self.canvas.update()

    def _apply_color_variance(self):
        if not self.canvas or not self.canvas.polygons:
            return
        variance = self._variance_slider.value()
        for polygon in self.canvas.polygons:
            c = polygon.get('color')
            if c is None or c.alpha() == 0:
                continue
            if c.red() == c.green() == c.blue():
                delta = random.randint(-variance, variance)
                v = max(0, min(255, c.red() + delta))
                polygon['color'] = QColor(v, v, v, c.alpha())
            else:
                r = max(0, min(255, c.red()   + random.randint(-variance, variance)))
                g = max(0, min(255, c.green() + random.randint(-variance, variance)))
                b = max(0, min(255, c.blue()  + random.randint(-variance, variance)))
                polygon['color'] = QColor(r, g, b, c.alpha())
        self.canvas.update()

    def _start_eyedropper(self):
        if self.canvas:
            self.canvas.eyedropper_mode = True
            self.canvas.setCursor(Qt.CrossCursor)
        self._eyedropper_btn.setChecked(True)
        self._eyedropper_btn.setStyleSheet('background-color: #f90; border: 2px solid #c60;')

    def _on_eyedropper_btn_clicked(self, checked):
        if checked:
            self._start_eyedropper()
        else:
            if self.canvas:
                self.canvas.eyedropper_mode = False
                self.canvas.setCursor(Qt.ArrowCursor)
            self._eyedropper_btn.setStyleSheet('')

    def receive_eyedropper_color(self, color):
        self.color_picker._apply_color(color)
        self.selected_color = color
        r, g, b = color.red(), color.green(), color.blue()
        self._chosen_color_box.setStyleSheet(
            f'background-color: rgb({r},{g},{b}); border: 2px solid #555;',
        )
        self._saved_palette.fill_armed_slot(color)
        self._eyedropper_btn.setChecked(False)
        self._eyedropper_btn.setStyleSheet('')

    def _on_palette_hover(self, color):
        if color is None:
            self._hover_preview.setStyleSheet('background-color: transparent; border: 1px solid #888;')
        else:
            r, g, b = color.red(), color.green(), color.blue()
            self._hover_preview.setStyleSheet(f'background-color: rgb({r},{g},{b}); border: 1px solid #888;')

    def _load_saved_color(self, color):
        self.color_picker._apply_color(color)
        self.selected_color = color
        r, g, b = color.red(), color.green(), color.blue()
        self._chosen_color_box.setStyleSheet(
            f'background-color: rgb({r},{g},{b}); border: 2px solid #555;',
        )

    def _on_picker_color_changed(self, color):
        self.selected_color = color
        r, g, b = color.red(), color.green(), color.blue()
        self._chosen_color_box.setStyleSheet(
            f'background-color: rgb({r},{g},{b}); border: 2px solid #555;',
        )

    def save_array(self):
        """Save polygons to CSV file compatible with mosaic_editor_pyqt"""
        if not self.canvas or not self.canvas.polygons:
            QMessageBox.warning(self, "Warning", "No polygons to save.")
            return
        
        # Open file dialog to choose save location
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Array as CSV",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not filename:
            return  # User cancelled
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header with alpha channel support (compatible with mosaic_editor_pyqt)
                writer.writerow(['polygon_id', 'coordinates', 'color_r', 'color_g', 'color_b', 'color_a'])
                
                # Write each polygon
                for i, polygon_data in enumerate(self.canvas.polygons):
                    points = polygon_data['points']
                    color = polygon_data['color']
                    
                    # Convert points to JSON string format (same as mosaic_editor_pyqt)
                    coords_json = json.dumps([[float(x), float(y)] for x, y in points])
                    
                    # Extract RGBA values (convert from QColor to 0-1 range)
                    r = color.red() / 255.0
                    g = color.green() / 255.0
                    b = color.blue() / 255.0
                    a = color.alpha() / 255.0
                    
                    # Write row
                    writer.writerow([i, coords_json, r, g, b, a])
            
            QMessageBox.information(
                self, 
                "Success", 
                f"Saved {len(self.canvas.polygons)} polygons to {filename}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save array: {str(e)}")
    
    def load_array(self):
        """Load polygons from CSV file compatible with mosaic_editor_pyqt"""
        if not self.canvas:
            return
            
        # Open file dialog to choose file
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Array from CSV",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not filename:
            return  # User cancelled
        
        try:
            polygons = []
            
            with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row_num, row in enumerate(reader, 1):
                    try:
                        # Parse coordinates - handle JSON array format
                        coords_str = row['coordinates'] if 'coordinates' in row else row.get('polygon_coords', '')
                        
                        # Remove quotes and parse as JSON
                        coords_str = coords_str.strip('"\'')
                        
                        try:
                            coord_list = json.loads(coords_str)
                            points = [(float(point[0]), float(point[1])) for point in coord_list]
                        except:
                            # Fallback to ast parsing for backward compatibility
                            import ast
                            coord_list = ast.literal_eval(coords_str)
                            points = [(float(point[0]), float(point[1])) for point in coord_list]
                        
                        if len(points) < 3:
                            continue
                        
                        # Parse color - handle separate R,G,B columns
                        if 'color_r' in row and 'color_g' in row and 'color_b' in row:
                            r = float(row['color_r'])
                            g = float(row['color_g'])
                            b = float(row['color_b'])
                            
                            # Check for alpha channel
                            if 'color_a' in row:
                                a = float(row['color_a'])
                                a = int(a * 255) if a <= 1.0 else int(a)
                            else:
                                a = 255  # Default to fully opaque
                            
                            # Convert from 0-1 range to 0-255
                            r = int(r * 255) if r <= 1.0 else int(r)
                            g = int(g * 255) if g <= 1.0 else int(g)
                            b = int(b * 255) if b <= 1.0 else int(b)
                            
                            color = QColor(r, g, b, a)
                        else:
                            # Default color if no color data
                            color = QColor(100, 100, 100)
                        
                        # Create polygon data structure
                        polygon_data = {
                            'points': points,
                            'color': color
                        }
                        polygons.append(polygon_data)
                        
                    except Exception as e:
                        print(f"Error parsing row {row_num}: {e}")
                        continue
            
            if polygons:
                # Clear existing polygons and load new ones
                self.canvas.polygons = polygons
                self.canvas.update()

                QMessageBox.information(
                    self,
                    "Success",
                    f"Loaded {len(polygons)} polygons from {filename}"
                )
            else:
                QMessageBox.warning(self, "Warning", "No valid polygons found in the file.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load array: {str(e)}")

    # ── Project save / load ───────────────────────────────────────────────
    # Captures the full canvas state into a single .mmp (Mandala Mosaic
    # Project) file: background image, polygons + their group bookkeeping,
    # mandala settings, circle settings, view.

    PROJECT_FORMAT_VERSION = 1

    @staticmethod
    def _pixmap_to_png_bytes(pm):
        if pm is None or pm.isNull():
            return None
        ba = QByteArray()
        buf = QBuffer(ba)
        buf.open(QBuffer.WriteOnly)
        pm.save(buf, "PNG")
        buf.close()
        return bytes(ba)

    @staticmethod
    def _png_bytes_to_pixmap(data):
        if not data:
            return None
        pm = QPixmap()
        pm.loadFromData(data, "PNG")
        return pm if not pm.isNull() else None

    def save_project(self):
        """Pickle the full canvas state to a .mmp file."""
        if not self.canvas:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "project.mmp",
            "Mandala Mosaic Project (*.mmp);;All files (*)",
        )
        if not path:
            return

        canvas = self.canvas

        # Pickle handles QColor natively and preserves shared references between
        # canvas.polygons, canvas.polygon_groups, and the parent_shape dicts,
        # so groups stay intact (each polygon in a group still points at the
        # same parent_shape dict after a roundtrip).
        data = {
            'format_version': self.PROJECT_FORMAT_VERSION,
            # Background image
            'background_png': self._pixmap_to_png_bytes(canvas.background_image),
            'show_image':     canvas.show_image,
            'image_offset_x': canvas.image_offset_x,
            'image_offset_y': canvas.image_offset_y,
            # Mandala settings
            'mandala_mode':                 canvas.mandala_mode,
            'num_copies':                   canvas.num_copies,
            'mandala_center_world_x':       canvas.mandala_center_world_x,
            'mandala_center_world_y':       canvas.mandala_center_world_y,
            'center_offset_x':              canvas.center_offset_x,
            'center_offset_y':              canvas.center_offset_y,
            # Circle settings
            'show_circle':            canvas.show_circle,
            'circle_diameter':        canvas.circle_diameter,
            'outer_circle_diameter':  canvas.outer_circle_diameter,
            # Polygons + group bookkeeping
            'polygons':         canvas.polygons,
            'polygon_groups':   canvas.polygon_groups,
            'current_group_id': canvas.current_group_id,
            # View
            'zoom_factor':  canvas.zoom_factor,
            'pan_offset_x': canvas.pan_offset_x,
            'pan_offset_y': canvas.pan_offset_y,
        }

        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save project: {e}")
            return
        QMessageBox.information(
            self, "Success",
            f"Project saved.\n\nFile: {path}\nPolygons: {len(canvas.polygons)}",
        )

    def load_project(self):
        """Restore the full canvas state from a .mmp file."""
        if not self.canvas:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Project", "",
            "Mandala Mosaic Project (*.mmp);;All files (*)",
        )
        if not path:
            return
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load project: {e}")
            return
        if not isinstance(data, dict):
            QMessageBox.critical(self, "Error", "File is not a valid Mandala Mosaic project.")
            return

        canvas = self.canvas

        # Background
        canvas.background_image = self._png_bytes_to_pixmap(data.get('background_png'))
        canvas.show_image     = data.get('show_image', True)
        canvas.image_offset_x = data.get('image_offset_x', 0)
        canvas.image_offset_y = data.get('image_offset_y', 0)

        # Mandala settings
        canvas.mandala_mode           = data.get('mandala_mode', True)
        canvas.num_copies             = data.get('num_copies', 6)
        canvas.mandala_center_world_x = data.get('mandala_center_world_x')
        canvas.mandala_center_world_y = data.get('mandala_center_world_y')
        canvas.center_offset_x        = data.get('center_offset_x', 0)
        canvas.center_offset_y        = data.get('center_offset_y', 0)

        # Circle settings
        canvas.show_circle            = data.get('show_circle', False)
        canvas.circle_diameter        = data.get('circle_diameter', 1000)
        canvas.outer_circle_diameter  = data.get('outer_circle_diameter', 1500)

        # Polygons + group bookkeeping (pickle preserved the shared references
        # between polygons[i] and polygon_groups[g]['polygons'][k], so editing
        # one still affects the other after load).
        canvas.polygons         = data.get('polygons', [])
        canvas.polygon_groups   = data.get('polygon_groups', [])
        canvas.current_group_id = data.get('current_group_id', 0)

        # View
        canvas.zoom_factor  = data.get('zoom_factor', 1.0)
        canvas.pan_offset_x = data.get('pan_offset_x', 0.0)
        canvas.pan_offset_y = data.get('pan_offset_y', 0.0)

        # Reset transient state — these never belong in a saved project.
        canvas.polygon_mode             = False
        canvas.polygon_points           = []
        canvas.eraser_mode              = False
        canvas.is_erasing               = False
        canvas.is_panning               = False
        canvas.last_pan_point           = None
        canvas.is_dragging_center       = False
        canvas.is_dragging_image        = False
        canvas.is_dragging_control_point = False
        canvas.selected_polygon_index   = -1
        canvas.selected_polygon_indices = []
        canvas.selected_control_point   = -1
        canvas.debug_circle_dots        = []

        # Reflect loaded values in the side-panel inputs (without firing the
        # handlers that would write back into the canvas).
        for attr, value, fmt in (
            ('copies_input',                 canvas.num_copies,            '{:d}'),
            ('circle_diameter_input',        canvas.circle_diameter,       '{:g}'),
            ('outer_circle_diameter_input',  canvas.outer_circle_diameter, '{:g}'),
        ):
            w = getattr(self, attr, None)
            if w is not None:
                w.blockSignals(True)
                try:
                    w.setText(fmt.format(value))
                except Exception:
                    w.setText(str(value))
                w.blockSignals(False)

        for chk_attr, val in (
            ('circle_checkbox',     canvas.show_circle),
            ('show_image_checkbox', canvas.show_image),
            ('mandala_checkbox',    canvas.mandala_mode),
            ('eraser_checkbox',     canvas.eraser_mode),
        ):
            cb = getattr(self, chk_attr, None)
            if cb is not None:
                cb.blockSignals(True)
                cb.setChecked(val)
                cb.blockSignals(False)

        canvas.update()
        QMessageBox.information(
            self, "Success",
            f"Project loaded.\n\nFile: {path}\nPolygons: {len(canvas.polygons)}",
        )


class MandalaMosaicWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        """Initialize the user interface"""
        self.setWindowTitle("Mandala Mosaic")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create central canvas first so the left panel can hook into it
        canvas = Canvas()

        # Create left panel with color tooling — needs the canvas reference so
        # the picker can paint / eyedrop / replace on it.
        left_panel = SidePanel("Left Panel", canvas)
        main_layout.addWidget(left_panel)
        main_layout.addWidget(canvas, 1)  # Give canvas stretch factor of 1

        # Create right panel (with reference to canvas for background loading)
        right_panel = SidePanel("Right Panel", canvas)
        main_layout.addWidget(right_panel)

        # Cross-references so canvas can call back into both panels (eyedropper
        # results, replace-source results, etc.). Matches duplicator.py.
        canvas.left_panel  = left_panel
        canvas.right_panel = right_panel

        # Global Ctrl+Z shortcut → undo the last polygon mutation. Attached
        # to the main window so it fires regardless of which widget has
        # focus (canvas, sidebar spinbox, palette, ...).
        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence
        undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        undo_shortcut.activated.connect(canvas.undo_last)

        central_widget.setLayout(main_layout)


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Create and show the main window
    window = MandalaMosaicWindow()
    window.show()
    
    # Start the event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
