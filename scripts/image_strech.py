import sys
import cv2
import numpy as np
import copy
import pickle
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QPushButton, QLabel, QFileDialog, QMessageBox, QScrollArea, QSpinBox, QDoubleSpinBox,
                             QSlider, QGroupBox, QFormLayout, QColorDialog)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QPoint, QPointF, QEvent, pyqtSignal, QRectF

class ImageCanvas(QWidget):
    selection_changed = pyqtSignal(int) # Emits index of selected polygon, or -1

    def __init__(self):
        super().__init__()
        self.image = None  # QImage
        self.cv_image = None # numpy array (RGB)
        self.display_image = None # numpy array (RGB) with effects applied
        self.points = []
        self.selecting_mode = False
        self.scale_factor = 1.0
        self.scroll_area = None
        self.target_width = 300
        self.target_height = 300
        self.polygons = []
        self.polygon_effects = [] # List of dicts for effects
        self.current_polygon = []
        self.drawing_polygon = False
        self.selected_polygon_index = None
        self.dragging_point_index = None
        self.image_tilt = 0  # Tilt angle for the entire image (horizontal)
        self.vertical_tilt = 0  # Tilt angle for vertical middle axis
        self.global_sharpness = 0  # Sharpness applied to entire image
        self.stretch_drag_start = None  # Starting point for stretch rectangle drag
        self.stretch_drag_current = None  # Current point during stretch rectangle drag
        self.show_grid = False  # Whether to display grid
        self.grid_size_percent = 10  # Grid cell size as percentage of image size
        self.grid_offset_x = 0  # Grid horizontal offset in pixels
        self.grid_offset_y = 0  # Grid vertical offset in pixels
        self.grid_line_thickness = 2  # Grid line thickness in pixels
        self.polygon_line_thickness = 1  # Polygon line thickness in pixels
        # Circles: each entry is [cx, cy, radius]
        self.circles = []
        self.drawing_circle = False
        self.circle_drag_start = None    # (cx, cy) fixed center while drawing
        self.circle_drag_current = None  # live mouse position while drawing
        self.selected_circle_index = None
        self.dragging_circle_index = None   # moving a circle by its center
        self.dragging_circle_offset = (0, 0)
        self.resizing_circle_index = None   # resizing by dragging the edge
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(400, 400)

    def set_target_resolution(self, width, height):
        self.target_width = width
        self.target_height = height

    def set_scroll_area(self, scroll_area):
        self.scroll_area = scroll_area

    def load_image(self, file_path):
        # Load image using OpenCV
        img = cv2.imread(file_path)
        if img is not None:
            new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # If a project image is already loaded (has polygons/circles), resize the new
            # image to match the existing image dimensions so all overlays stay aligned.
            if self.cv_image is not None and (self.polygons or self.circles):
                old_h, old_w = self.cv_image.shape[:2]
                new_img = cv2.resize(new_img, (old_w, old_h), interpolation=cv2.INTER_LANCZOS4)
            self.cv_image = new_img
            self.display_image = self.cv_image.copy()
            self.scale_factor = 1.0
            self.current_polygon = []
            self.selecting_mode = False
            self.drawing_polygon = False
            self.drawing_circle = False
            self.selected_circle_index = None
            self.dragging_circle_index = None
            self.resizing_circle_index = None
            self.apply_effects()
            self.update()
        else:
            QMessageBox.critical(self, "Error", "Failed to load image.")

    def _compute_image_pan(self):
        """Return (pan_x, pan_y) in image pixels: free pan without wrapping."""
        return float(self.grid_offset_x), float(self.grid_offset_y)

    def update_canvas_size(self):
        """Resize the widget to fit the image. Pan does not change the widget size."""
        if self.image is None:
            return
        label_margin = 30
        self.setFixedSize(
            int((self.image.width()  + label_margin) * self.scale_factor),
            int((self.image.height() + label_margin) * self.scale_factor)
        )

    def update_image_from_cv(self):
        if self.display_image is None:
            return
        # Ensure the array is C-contiguous so bytes_per_line calculation is correct
        img = np.ascontiguousarray(self.display_image)
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        # .copy() makes the QImage own its data, preventing stale-buffer bugs
        self.image = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
        self.update_canvas_size()
        self.update()

    def apply_effects(self):
        if self.cv_image is None:
            return

        # Apply tilt to the entire image first
        if self.image_tilt != 0 or self.vertical_tilt != 0:
            img_h, img_w = self.cv_image.shape[:2]
            
            # Source points (corners of image)
            src_pts = np.float32([
                [0, 0],
                [img_w, 0],
                [img_w, img_h],
                [0, img_h]
            ])
            
            # Start with original corners
            dst_pts = np.float32([
                [0, 0],
                [img_w, 0],
                [img_w, img_h],
                [0, img_h]
            ])
            
            # Apply horizontal tilt (image_tilt)
            if self.image_tilt != 0:
                angle_rad = np.deg2rad(self.image_tilt)
                offset = img_w * np.tan(angle_rad) * 0.5
                
                dst_pts = np.float32([
                    [max(0, offset), 0],
                    [min(img_w, img_w - offset), 0],
                    [min(img_w, img_w + offset), img_h],
                    [max(0, -offset), img_h]
                ])
            
            # Apply vertical tilt (vertical_tilt)
            if self.vertical_tilt != 0:
                angle_rad = np.deg2rad(self.vertical_tilt)
                offset = img_h * np.tan(angle_rad) * 0.5
                
                # Adjust y-coordinates for vertical tilt
                # Top edge shifts, bottom edge shifts opposite
                dst_pts = np.float32([
                    [dst_pts[0][0], max(0, -offset)],
                    [dst_pts[1][0], max(0, offset)],
                    [dst_pts[2][0], min(img_h, img_h - offset)],
                    [dst_pts[3][0], min(img_h, img_h + offset)]
                ])
            
            # Get perspective transform matrix
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            # Apply perspective transformation
            self.display_image = cv2.warpPerspective(self.cv_image, M, (img_w, img_h), 
                                                     borderMode=cv2.BORDER_CONSTANT, 
                                                     borderValue=(0, 0, 0))
        else:
            self.display_image = self.cv_image.copy()
        
        for i, poly in enumerate(self.polygons):
            if i >= len(self.polygon_effects):
                continue
                
            effects = self.polygon_effects[i]
            
            # Create mask for polygon
            mask = np.zeros(self.cv_image.shape[:2], dtype=np.uint8)
            pts = np.array(poly, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            
            # Extract ROI
            # We can optimize by bounding rect, but for now full image masking is easier to implement
            
            # 1. Brightness and Contrast
            # alpha = contrast (1.0 is original), beta = brightness (0 is original)
            alpha = effects.get('contrast', 1.0)
            beta = effects.get('brightness', 0)
            black_point = effects.get('black_point', 0)
            
            # 2. Saturation
            sat_scale = effects.get('saturation', 1.0)
            
            # 3. Warmth (Temperature)
            warmth = effects.get('warmth', 0)

            # 4. Tint
            tint_color = effects.get('tint_color', (255, 255, 255)) # RGB
            tint_strength = effects.get('tint_strength', 0) / 100.0 # 0.0 to 1.0
            
            # Apply to the whole image (or ROI) then mask copy back
            # To avoid processing full image, let's crop to bounding rect
            x, y, w, h = cv2.boundingRect(pts)
            roi = self.display_image[y:y+h, x:x+w].astype(np.float32)
            roi_mask = mask[y:y+h, x:x+w]
            
            # Only process if mask is not empty
            if np.sum(roi_mask) > 0:
                # Black Point - crush blacks
                if black_point > 0:
                    roi = np.maximum(0, roi - black_point)
                
                # Brightness/Contrast
                roi = roi * alpha + beta
                roi = np.clip(roi, 0, 255)
                
                # Saturation
                if sat_scale != 1.0:
                    roi_hsv = cv2.cvtColor(roi.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                    roi_hsv[:, :, 1] *= sat_scale
                    roi_hsv[:, :, 1] = np.clip(roi_hsv[:, :, 1], 0, 255)
                    roi = cv2.cvtColor(roi_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

                # Warmth (R increases, B decreases)
                if warmth != 0:
                    # R channel is 0, B is 2 in RGB
                    r = roi[:, :, 0]
                    b = roi[:, :, 2]
                    
                    r += warmth
                    b -= warmth
                    
                    roi[:, :, 0] = np.clip(r, 0, 255)
                    roi[:, :, 2] = np.clip(b, 0, 255)

                # Tint
                if tint_strength > 0:
                    # Create a solid color layer
                    tint_layer = np.full_like(roi, tint_color, dtype=np.float32)
                    # Blend
                    roi = cv2.addWeighted(roi, 1.0 - tint_strength, tint_layer, tint_strength, 0)

                # Blend back
                roi = roi.astype(np.uint8)
                
                # Use mask to copy only polygon area
                # We need 3-channel mask
                roi_mask_3 = cv2.merge([roi_mask, roi_mask, roi_mask])
                
                # Where mask is set, use processed ROI, else use original (which is already in display_image)
                # Actually we are modifying display_image in place
                current_roi = self.display_image[y:y+h, x:x+w]
                np.copyto(current_roi, roi, where=roi_mask_3.astype(bool))

        # Apply global sharpness to the entire image
        if self.global_sharpness > 0:
            blurred = cv2.GaussianBlur(self.display_image, (0, 0), sigmaX=3)
            strength = self.global_sharpness / 50.0  # 0.0 to 2.0
            self.display_image = cv2.addWeighted(self.display_image, 1.0 + strength, blurred, -strength, 0)

        self.update_image_from_cv()

    def start_stretch_selection(self):
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return
        self.selecting_mode = True
        self.drawing_polygon = False
        self.drawing_circle = False
        self.circle_drag_start = None
        self.circle_drag_current = None
        self.points = []
        self.stretch_drag_start = None
        self.stretch_drag_current = None
        self.update()

    def start_polygon_drawing(self):
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return False
        self.drawing_polygon = True
        self.selecting_mode = False
        self.drawing_circle = False
        self.circle_drag_start = None
        self.circle_drag_current = None
        self.current_polygon = []
        self.update()
        return True

    def stop_polygon_drawing(self):
        self.drawing_polygon = False
        self.current_polygon = []
        self.update()

    def start_circle_drawing(self):
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return False
        self.drawing_circle = True
        self.drawing_polygon = False
        self.selecting_mode = False
        self.circle_drag_start = None
        self.circle_drag_current = None
        self.update()
        return True

    def stop_circle_drawing(self):
        self.drawing_circle = False
        self.circle_drag_start = None
        self.circle_drag_current = None
        self.update()

    def mousePressEvent(self, event):
        if self.image:
            pos = event.pos()
            # Account for label offset (30 pixels for grid labels)
            label_offset = 30
            # Convert to image coordinates
            img_x = (pos.x() - label_offset * self.scale_factor) / self.scale_factor
            img_y = (pos.y() - label_offset * self.scale_factor) / self.scale_factor
            
            # Ensure point is within image bounds (allow slightly outside for editing handles)
            if 0 <= img_x < self.image.width() and 0 <= img_y < self.image.height() or (not self.selecting_mode and not self.drawing_polygon and not self.drawing_circle):
                
                if self.selecting_mode:
                    # Start rectangle drag
                    self.stretch_drag_start = (img_x, img_y)
                    self.stretch_drag_current = (img_x, img_y)
                    self.update()

                elif self.drawing_circle:
                    # Left-click begins a new circle drag
                    if event.button() == Qt.LeftButton:
                        self.circle_drag_start = (img_x, img_y)
                        self.circle_drag_current = (img_x, img_y)
                        self.update()

                elif self.drawing_polygon:
                    if event.button() == Qt.LeftButton:
                        self.current_polygon.append((img_x, img_y))
                        self.update()
                    elif event.button() == Qt.RightButton:
                        if len(self.current_polygon) > 2:
                            self.polygons.append(self.current_polygon)
                            # Add default effects for new polygon
                            self.polygon_effects.append({
                                'brightness': 0,
                                'contrast': 1.0,
                                'saturation': 1.0,
                                'warmth': 0,
                                'tint_color': (255, 255, 255),
                                'tint_strength': 0,
                                'black_point': 0,
                                'sharpness': 0
                            })
                            self.current_polygon = []
                            # self.drawing_polygon = False # Keep drawing mode active
                            self.update()
                        else:
                            # Maybe cancel if not enough points? Or just ignore
                            pass
                else:
                    # Edit mode
                    self.handle_edit_click(img_x, img_y)

    def handle_edit_click(self, img_x, img_y):
        hit_radius = 10 / self.scale_factor
        found_hit = False
        old_selection = self.selected_polygon_index
        
        # 1. Check vertices of currently selected polygon
        if self.selected_polygon_index is not None:
            poly = self.polygons[self.selected_polygon_index]
            for i, pt in enumerate(poly):
                if (pt[0] - img_x)**2 + (pt[1] - img_y)**2 < hit_radius**2:
                    self.dragging_point_index = i
                    found_hit = True
                    break
        
        # 2. Check vertices of all polygons (switch selection)
        if not found_hit:
            for p_idx, poly in enumerate(self.polygons):
                for i, pt in enumerate(poly):
                    if (pt[0] - img_x)**2 + (pt[1] - img_y)**2 < hit_radius**2:
                        self.selected_polygon_index = p_idx
                        self.dragging_point_index = i
                        found_hit = True
                        break
                if found_hit: break
        
        # 3. Check inside polygons
        if not found_hit:
            for p_idx, poly in enumerate(self.polygons):
                pts_np = np.array(poly, dtype=np.int32)
                dist = cv2.pointPolygonTest(pts_np, (img_x, img_y), False)
                if dist >= 0:
                    self.selected_polygon_index = p_idx
                    self.dragging_point_index = None
                    found_hit = True
                    break
        
        # 4. Check circles: centre hit → move, edge hit → resize
        if not found_hit:
            hit_r = 12 / self.scale_factor
            for c_idx, circ in enumerate(self.circles):
                cx, cy, radius = circ
                dist = ((img_x - cx) ** 2 + (img_y - cy) ** 2) ** 0.5
                if dist <= hit_r:                     # centre
                    self.selected_circle_index = c_idx
                    self.selected_polygon_index = None
                    self.dragging_circle_index = c_idx
                    self.dragging_circle_offset = (img_x - cx, img_y - cy)
                    self.resizing_circle_index = None
                    self.dragging_point_index = None
                    found_hit = True
                    break
                elif abs(dist - radius) <= hit_r:     # edge
                    self.selected_circle_index = c_idx
                    self.selected_polygon_index = None
                    self.resizing_circle_index = c_idx
                    self.dragging_circle_index = None
                    self.dragging_point_index = None
                    found_hit = True
                    break

        # 5. Deselect if clicked empty space
        if not found_hit:
            self.selected_polygon_index = None
            self.selected_circle_index = None
            self.dragging_point_index = None
            self.dragging_circle_index = None
            self.resizing_circle_index = None

        if self.selected_polygon_index != old_selection:
            self.selection_changed.emit(self.selected_polygon_index if self.selected_polygon_index is not None else -1)

        self.update()

    def mouseMoveEvent(self, event):
        if self.selecting_mode and self.stretch_drag_start is not None:
            # Update current drag position for stretch rectangle
            pos = event.pos()
            label_offset = 30
            img_x = (pos.x() - label_offset * self.scale_factor) / self.scale_factor
            img_y = (pos.y() - label_offset * self.scale_factor) / self.scale_factor
            
            # Constrain to square
            start_x, start_y = self.stretch_drag_start
            dx = img_x - start_x
            dy = img_y - start_y
            
            # Use the larger dimension to create a square
            size = max(abs(dx), abs(dy))
            # Maintain direction
            square_x = start_x + (size if dx >= 0 else -size)
            square_y = start_y + (size if dy >= 0 else -size)
            
            self.stretch_drag_current = (square_x, square_y)
            self.update()
        elif self.drawing_circle and self.circle_drag_start is not None:
            # Live preview while drawing a new circle
            pos = event.pos()
            label_offset = 30
            img_x = (pos.x() - label_offset * self.scale_factor) / self.scale_factor
            img_y = (pos.y() - label_offset * self.scale_factor) / self.scale_factor
            self.circle_drag_current = (img_x, img_y)
            self.update()
        elif self.dragging_circle_index is not None:
            # Moving an existing circle
            pos = event.pos()
            label_offset = 30
            img_x = (pos.x() - label_offset * self.scale_factor) / self.scale_factor
            img_y = (pos.y() - label_offset * self.scale_factor) / self.scale_factor
            ox, oy = self.dragging_circle_offset
            _, _, radius = self.circles[self.dragging_circle_index]
            self.circles[self.dragging_circle_index] = [img_x - ox, img_y - oy, radius]
            self.update()
        elif self.resizing_circle_index is not None:
            # Resizing an existing circle by dragging its edge
            pos = event.pos()
            label_offset = 30
            img_x = (pos.x() - label_offset * self.scale_factor) / self.scale_factor
            img_y = (pos.y() - label_offset * self.scale_factor) / self.scale_factor
            cx, cy, _ = self.circles[self.resizing_circle_index]
            new_radius = ((img_x - cx) ** 2 + (img_y - cy) ** 2) ** 0.5
            if new_radius > 2:
                self.circles[self.resizing_circle_index] = [cx, cy, new_radius]
            self.update()
        elif self.dragging_point_index is not None and self.selected_polygon_index is not None:
             pos = event.pos()
             label_offset = 30
             img_x = (pos.x() - label_offset * self.scale_factor) / self.scale_factor
             img_y = (pos.y() - label_offset * self.scale_factor) / self.scale_factor
             
             # Update point
             self.polygons[self.selected_polygon_index][self.dragging_point_index] = (img_x, img_y)
             # Re-apply effects because polygon shape changed
             self.apply_effects()
             self.update()

    def mouseReleaseEvent(self, event):
        if self.selecting_mode and self.stretch_drag_start is not None and self.stretch_drag_current is not None:
            # Convert rectangle to 4 points and perform stretch
            x1, y1 = self.stretch_drag_start
            x2, y2 = self.stretch_drag_current
            
            # Create 4 corner points from the rectangle
            self.points = [
                (min(x1, x2), min(y1, y2)),  # top-left
                (max(x1, x2), min(y1, y2)),  # top-right
                (max(x1, x2), max(y1, y2)),  # bottom-right
                (min(x1, x2), max(y1, y2))   # bottom-left
            ]
            
            # Only perform stretch if rectangle has some size
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                self.perform_stretch()
            
            # Reset stretch mode
            self.selecting_mode = False
            self.stretch_drag_start = None
            self.stretch_drag_current = None
            self.update()
        elif self.drawing_circle and self.circle_drag_start is not None:
            # Commit the new circle
            cx, cy = self.circle_drag_start
            mx, my = self.circle_drag_current if self.circle_drag_current else self.circle_drag_start
            radius = ((mx - cx) ** 2 + (my - cy) ** 2) ** 0.5
            if radius > 3:
                self.circles.append([cx, cy, radius])
            self.circle_drag_start = None
            self.circle_drag_current = None
            self.update()
        else:
            self.dragging_point_index = None
            self.dragging_circle_index = None
            self.resizing_circle_index = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.selected_polygon_index = None
            self.selected_circle_index = None
            self.dragging_point_index = None
            self.dragging_circle_index = None
            self.resizing_circle_index = None
            self.selection_changed.emit(-1)
            self.update()
        elif event.key() == Qt.Key_Delete:
            if self.selected_polygon_index is not None:
                self.polygons.pop(self.selected_polygon_index)
                self.polygon_effects.pop(self.selected_polygon_index)
                self.selected_polygon_index = None
                self.dragging_point_index = None
                self.selection_changed.emit(-1)
                self.apply_effects()
                self.update()
            elif self.selected_circle_index is not None:
                self.circles.pop(self.selected_circle_index)
                self.selected_circle_index = None
                self.update()

    def perform_stretch(self):
        if len(self.points) != 4:
            return

        pts = np.array(self.points, dtype="float32")
        
        # Sort points to order: top-left, top-right, bottom-right, bottom-left
        rect = np.zeros((4, 2), dtype="float32")
        
        # Top-left will have the smallest sum, bottom-right will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Top-right will have the smallest difference, bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        # Destination points for target resolution
        w = self.target_width
        h = self.target_height
        dst = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]], dtype="float32")
            
        # Compute the perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(self.cv_image, M, (w, h))
        
        self.cv_image = warped
        self.display_image = self.cv_image.copy() # Update display image
        self.polygons = [] # Clear polygons as they don't match the new image
        self.polygon_effects = [] # Clear effects
        self.selected_polygon_index = None
        self.dragging_point_index = None
        self.selection_changed.emit(-1)
        
        self.scale_factor = 1.0 # Reset zoom after stretch
        self.update_image_from_cv()
        self.points = []
        QMessageBox.information(self, "Success", f"Image stretched to {w}x{h} pixels.")


    def handle_zoom(self, event, viewport_pos):
        if self.image:
            old_scale = self.scale_factor
            
            # Get current scrollbar values
            old_scroll_x = self.scroll_area.horizontalScrollBar().value()
            old_scroll_y = self.scroll_area.verticalScrollBar().value()

            delta = event.angleDelta().y()
            if delta > 0:
                new_scale = old_scale * 1.1
            else:
                new_scale = old_scale * 0.9
            
            # Limit zoom
            new_scale = max(0.1, min(new_scale, 10.0))
            
            self.scale_factor = new_scale
            self.update_image_from_cv()

            # Adjust scrollbars to zoom towards cursor
            if self.scroll_area:
                # viewport_pos is relative to the viewport
                # We need the position relative to the content (canvas) BEFORE the zoom
                # But since we just resized, the canvas coordinates have changed.
                
                # Let's use the viewport position directly.
                # The point under the cursor in the viewport should remain under the cursor.
                # Viewport X = (Content X - Scroll X)
                # Content X = Viewport X + Scroll X
                
                # We want: (New Content X - New Scroll X) = Viewport X
                # New Scroll X = New Content X - Viewport X
                
                # New Content X = Old Content X * (new_scale / old_scale)
                # Old Content X = Viewport X + Old Scroll X
                
                scale_ratio = new_scale / old_scale
                
                mouse_x_viewport = viewport_pos.x()
                mouse_y_viewport = viewport_pos.y()
                
                old_content_x = mouse_x_viewport + old_scroll_x
                old_content_y = mouse_y_viewport + old_scroll_y
                
                new_content_x = old_content_x * scale_ratio
                new_content_y = old_content_y * scale_ratio
                
                new_scroll_x = new_content_x - mouse_x_viewport
                new_scroll_y = new_content_y - mouse_y_viewport
                
                self.scroll_area.horizontalScrollBar().setValue(int(new_scroll_x))
                self.scroll_area.verticalScrollBar().setValue(int(new_scroll_y))

    def wheelEvent(self, event):
        # This might not be called if ScrollArea intercepts it, 
        # but we keep it for cases where it is called.
        # We'll use the event filter in MainWindow to ensure it works.
        pass

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # Define offset for grid labels
        label_offset = 30

        # Compute how much to pan the image (image moves, grid stays fixed)
        img_pan_x = float(self.grid_offset_x)
        img_pan_y = float(self.grid_offset_y)

        if self.image:
            # Draw image shifted so it pans under the fixed grid
            img_width = self.image.width()
            img_height = self.image.height()
            target_rect = QRectF(
                (label_offset + img_pan_x) * self.scale_factor,
                (label_offset + img_pan_y) * self.scale_factor,
                img_width * self.scale_factor,
                img_height * self.scale_factor)
            painter.drawImage(target_rect, self.image)
        
        # Translate and scale for drawing overlays
        painter.translate(label_offset * self.scale_factor, label_offset * self.scale_factor)
        painter.scale(self.scale_factor, self.scale_factor)
        
        # Draw grid if enabled
        if self.show_grid and self.image and self.grid_size_percent > 0:
            painter.setPen(QPen(QColor(173, 216, 230), self.grid_line_thickness))  # Light blue
            
            # Get image dimensions
            img_width = self.image.width()
            img_height = self.image.height()
            
            # Calculate grid cell size from percentage (use smaller dimension for square cells)
            smaller_dimension = min(img_width, img_height)
            grid_cell_size = smaller_dimension * (self.grid_size_percent / 100.0)
            
            # Grid origin is always fixed at (0,0) — image pans underneath
            # Draw vertical lines
            x = 0.0
            while x <= img_width:
                painter.drawLine(QPointF(x, 0), QPointF(x, img_height))
                x += grid_cell_size
            
            # Draw horizontal lines
            y = 0.0
            while y <= img_height:
                painter.drawLine(QPointF(0, y), QPointF(img_width, y))
                y += grid_cell_size
            
            # Draw grid labels
            painter.setPen(QPen(QColor(0, 0, 255), 1))  # Blue text
            font = QFont()
            font.setPixelSize(max(14, int(grid_cell_size / 6)))  # Scale font with grid size
            painter.setFont(font)
            
            # Calculate font metrics for better positioning
            font_height = painter.fontMetrics().height()
            
            # Draw column numbers at the top
            x = 0.0
            col_num = 1
            while x < img_width:
                center_x = x + grid_cell_size / 2
                text = str(col_num)
                text_width = painter.fontMetrics().horizontalAdvance(text)
                painter.drawText(int(center_x - text_width / 2), int(-5), text)
                x += grid_cell_size
                col_num += 1

            # Draw row letters on the left
            y = 0.0
            row_num = 0
            while y < img_height:
                center_y = y + grid_cell_size / 2
                text = chr(ord('A') + row_num)
                painter.drawText(int(-15), int(center_y + font_height / 3), text)
                y += grid_cell_size
                row_num += 1

        # Draw completed polygons
        if self.polygons:
            for idx, poly in enumerate(self.polygons):
                # Highlight selected polygon
                if idx == self.selected_polygon_index:
                    painter.setPen(QPen(Qt.magenta, self.polygon_line_thickness))
                else:
                    painter.setPen(QPen(Qt.green, self.polygon_line_thickness))
                
                if len(poly) > 1:
                    for i in range(len(poly) - 1):
                        painter.drawLine(QPointF(poly[i][0], poly[i][1]), 
                                         QPointF(poly[i+1][0], poly[i+1][1]))
                    # Close loop
                    painter.drawLine(QPointF(poly[-1][0], poly[-1][1]), 
                                     QPointF(poly[0][0], poly[0][1]))
                
                # Draw control points for selected polygon
                if idx == self.selected_polygon_index:
                    painter.setPen(QPen(Qt.magenta, 8))
                    for pt in poly:
                        painter.drawPoint(QPointF(pt[0], pt[1]))

        # Draw current polygon being drawn
        if self.drawing_polygon and self.current_polygon:
            painter.setPen(QPen(Qt.blue, 1))
            for i in range(len(self.current_polygon) - 1):
                painter.drawLine(QPointF(self.current_polygon[i][0], self.current_polygon[i][1]), 
                                 QPointF(self.current_polygon[i+1][0], self.current_polygon[i+1][1]))
            
            # Draw points
            painter.setPen(QPen(Qt.yellow, 5))
            for pt in self.current_polygon:
                painter.drawPoint(QPointF(pt[0], pt[1]))

        # Draw stretch rectangle during drag
        if self.selecting_mode and self.stretch_drag_start is not None and self.stretch_drag_current is not None:
            x1, y1 = self.stretch_drag_start
            x2, y2 = self.stretch_drag_current
            
            # Draw rectangle
            painter.setPen(QPen(Qt.yellow, 2))
            painter.drawRect(QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)))
            
            # Draw corner points
            painter.setPen(QPen(Qt.red, 8))
            painter.drawPoint(QPointF(x1, y1))
            painter.drawPoint(QPointF(x2, y2))

        # Draw completed circles
        for idx, circ in enumerate(self.circles):
            cx, cy, radius = circ
            selected = (idx == self.selected_circle_index)
            painter.setPen(QPen(Qt.magenta if selected else QColor(0, 200, 255), 3))
            painter.drawEllipse(QPointF(cx, cy), radius, radius)
            # Centre dot (larger when selected)
            painter.setPen(QPen(Qt.magenta if selected else QColor(0, 200, 255), 10 if selected else 6))
            painter.drawPoint(QPointF(cx, cy))

        # Draw circle preview while dragging a new one
        if self.drawing_circle and self.circle_drag_start is not None and self.circle_drag_current is not None:
            cx, cy = self.circle_drag_start
            mx, my = self.circle_drag_current
            radius = ((mx - cx) ** 2 + (my - cy) ** 2) ** 0.5
            painter.setPen(QPen(Qt.blue, 2, Qt.DashLine))
            painter.drawEllipse(QPointF(cx, cy), radius, radius)
            painter.setPen(QPen(Qt.yellow, 9))
            painter.drawPoint(QPointF(cx, cy))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Stretch Editor")
        self.resize(800, 600)
        
        self.canvas = ImageCanvas()
        
        # Scroll area for the canvas
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.canvas)
        self.scroll_area.setWidgetResizable(False) # Honor the widget's size (important for scrolling large images)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        
        # Pass scroll area to canvas for zoom handling
        self.canvas.set_scroll_area(self.scroll_area)
        
        # Install event filter to capture wheel events for zooming
        self.scroll_area.viewport().installEventFilter(self)

        self.copied_effects = None

        # Layouts
        main_layout = QHBoxLayout()
        
        # Sidebar container
        sidebar_widget = QWidget()
        sidebar_widget.setFixedWidth(300)
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(0, 0, 0, 0) # Remove extra margins
        
        # Buttons
        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)
        
        self.stretch_btn = QPushButton("Stretch")
        self.stretch_btn.clicked.connect(self.start_stretch_mode)
        self.stretch_btn.setToolTip("Click 4 points on the image to define corners")
        
        self.polygon_btn = QPushButton("Polygon")
        self.polygon_btn.setCheckable(True)
        self.polygon_btn.clicked.connect(self.toggle_polygon_mode)
        self.polygon_btn.setToolTip("Left click to add points, Right click to finish polygon")

        self.circle_btn = QPushButton("Circle")
        self.circle_btn.setCheckable(True)
        self.circle_btn.clicked.connect(self.toggle_circle_mode)
        self.circle_btn.setToolTip("Drag to draw a circle. Then click centre to move, edge to resize, Delete to remove.")

        save_btn = QPushButton("Save Image")
        save_btn.clicked.connect(self.save_image)

        save_circle_btn = QPushButton("Save Circle")
        save_circle_btn.clicked.connect(self.save_circle_image)
        save_circle_btn.setToolTip("Save the image inside the selected circle at full original resolution")
        
        save_array_btn = QPushButton("Save Array")
        save_array_btn.clicked.connect(self.save_array)
        
        load_array_btn = QPushButton("Load Array")
        load_array_btn.clicked.connect(self.load_array)
        
        save_project_btn = QPushButton("Save Project")
        save_project_btn.clicked.connect(self.save_project)
        
        load_project_btn = QPushButton("Load Project")
        load_project_btn.clicked.connect(self.load_project)
        
        # Resolution inputs
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 20000)
        self.width_spin.setValue(300)
        self.width_spin.setSuffix(" px")
        self.width_spin.setToolTip("Target Width")
        self.width_spin.valueChanged.connect(self.update_resolution)
        
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 20000)
        self.height_spin.setValue(300)
        self.height_spin.setSuffix(" px")
        self.height_spin.setToolTip("Target Height")
        self.height_spin.valueChanged.connect(self.update_resolution)

        # Add widgets to sidebar
        sidebar_layout.addWidget(load_btn)
        sidebar_layout.addWidget(self.stretch_btn)
        sidebar_layout.addWidget(self.polygon_btn)
        sidebar_layout.addWidget(self.circle_btn)
        sidebar_layout.addWidget(save_btn)
        sidebar_layout.addWidget(save_circle_btn)
        sidebar_layout.addWidget(save_array_btn)
        sidebar_layout.addWidget(load_array_btn)
        sidebar_layout.addWidget(save_project_btn)
        sidebar_layout.addWidget(load_project_btn)
        
        sidebar_layout.addSpacing(20)
        sidebar_layout.addWidget(QLabel("Target Size:"))
        
        size_layout = QHBoxLayout()
        size_layout.addWidget(self.width_spin)
        size_layout.addWidget(QLabel("x"))
        size_layout.addWidget(self.height_spin)
        sidebar_layout.addLayout(size_layout)

        sidebar_layout.addSpacing(10)
        sidebar_layout.addWidget(QLabel("Scale Image (%):" ))

        self.img_width_scale_spin = QDoubleSpinBox()
        self.img_width_scale_spin.setRange(0.1, 10000.0)
        self.img_width_scale_spin.setDecimals(1)
        self.img_width_scale_spin.setSingleStep(0.1)
        self.img_width_scale_spin.setValue(100.0)
        self.img_width_scale_spin.setSuffix(" % W")
        self.img_width_scale_spin.setToolTip("Scale image width by this percentage")

        self.img_height_scale_spin = QDoubleSpinBox()
        self.img_height_scale_spin.setRange(0.1, 10000.0)
        self.img_height_scale_spin.setDecimals(1)
        self.img_height_scale_spin.setSingleStep(0.1)
        self.img_height_scale_spin.setValue(100.0)
        self.img_height_scale_spin.setSuffix(" % H")
        self.img_height_scale_spin.setToolTip("Scale image height by this percentage")

        img_scale_layout = QHBoxLayout()
        img_scale_layout.addWidget(self.img_width_scale_spin)
        img_scale_layout.addWidget(self.img_height_scale_spin)
        sidebar_layout.addLayout(img_scale_layout)

        apply_img_scale_btn = QPushButton("Apply Image Scale")
        apply_img_scale_btn.setToolTip("Resize the image by the given width/height percentages")
        apply_img_scale_btn.clicked.connect(self.apply_image_scale)
        sidebar_layout.addWidget(apply_img_scale_btn)

        sidebar_layout.addSpacing(20)
        
        # Image Tilt Control (applies to entire image)
        tilt_group = QGroupBox("Image Tilt")
        tilt_layout = QFormLayout()
        self.image_tilt_slider = QSlider(Qt.Horizontal)
        self.image_tilt_slider.setRange(-45, 45)
        self.image_tilt_slider.setValue(0)
        self.image_tilt_slider.valueChanged.connect(self.update_image_tilt)
        tilt_layout.addRow("Horizontal Tilt", self.image_tilt_slider)
        
        self.vertical_tilt_slider = QSlider(Qt.Horizontal)
        self.vertical_tilt_slider.setRange(-45, 45)
        self.vertical_tilt_slider.setValue(0)
        self.vertical_tilt_slider.valueChanged.connect(self.update_vertical_tilt)
        tilt_layout.addRow("Vertical Tilt", self.vertical_tilt_slider)
        
        tilt_layout.addRow("Sharpness", self._make_sharpness_slider())
        tilt_group.setLayout(tilt_layout)
        sidebar_layout.addWidget(tilt_group)
        
        sidebar_layout.addSpacing(20)
        
        # Polygon Effects Group
        self.effects_group = QGroupBox("Polygon Effects")
        self.effects_group.setEnabled(False)
        effects_layout = QFormLayout()
        
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.update_effects)
        
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 200) # 0.0 to 2.0 (div by 100)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self.update_effects)
        
        self.saturation_slider = QSlider(Qt.Horizontal)
        self.saturation_slider.setRange(0, 200) # 0.0 to 2.0 (div by 100)
        self.saturation_slider.setValue(100)
        self.saturation_slider.valueChanged.connect(self.update_effects)
        
        self.warmth_slider = QSlider(Qt.Horizontal)
        self.warmth_slider.setRange(-100, 100)
        self.warmth_slider.setValue(0)
        self.warmth_slider.valueChanged.connect(self.update_effects)
        
        self.black_point_slider = QSlider(Qt.Horizontal)
        self.black_point_slider.setRange(0, 100)
        self.black_point_slider.setValue(0)
        self.black_point_slider.valueChanged.connect(self.update_effects)

        self.tint_btn = QPushButton("Select Tint Color")
        self.tint_btn.clicked.connect(self.select_tint_color)
        self.tint_btn.setStyleSheet("background-color: white; color: black;")
        
        self.tint_strength_slider = QSlider(Qt.Horizontal)
        self.tint_strength_slider.setRange(0, 100)
        self.tint_strength_slider.setValue(0)
        self.tint_strength_slider.valueChanged.connect(self.update_effects)

        # Copy/Paste Buttons
        copy_paste_layout = QHBoxLayout()
        self.copy_btn = QPushButton("Copy Colors")
        self.copy_btn.clicked.connect(self.copy_colors)
        self.paste_btn = QPushButton("Paste Colors")
        self.paste_btn.clicked.connect(self.paste_colors)
        self.paste_btn.setEnabled(False)
        
        copy_paste_layout.addWidget(self.copy_btn)
        copy_paste_layout.addWidget(self.paste_btn)

        effects_layout.addRow(copy_paste_layout)
        effects_layout.addRow("Brightness", self.brightness_slider)
        effects_layout.addRow("Contrast", self.contrast_slider)
        effects_layout.addRow("Saturation", self.saturation_slider)
        effects_layout.addRow("Warmth", self.warmth_slider)
        effects_layout.addRow("Black Point", self.black_point_slider)
        effects_layout.addRow("Tint Color", self.tint_btn)
        effects_layout.addRow("Tint Strength", self.tint_strength_slider)
        
        self.effects_group.setLayout(effects_layout)
        sidebar_layout.addWidget(self.effects_group)
        
        sidebar_layout.addStretch() # Push items to top
        
        # Right sidebar container
        right_sidebar_widget = QWidget()
        right_sidebar_widget.setFixedWidth(250)
        right_sidebar_layout = QVBoxLayout(right_sidebar_widget)
        right_sidebar_layout.setContentsMargins(0, 0, 0, 0)
        
        # Grid controls
        right_sidebar_layout.addWidget(QLabel("<b>Grid Controls</b>"))
        right_sidebar_layout.addSpacing(10)
        
        self.grid_btn = QPushButton("Grid")
        self.grid_btn.setCheckable(True)
        self.grid_btn.clicked.connect(self.toggle_grid)
        self.grid_btn.setToolTip("Toggle grid display")
        right_sidebar_layout.addWidget(self.grid_btn)
        
        right_sidebar_layout.addSpacing(10)
        right_sidebar_layout.addWidget(QLabel("Grid Size:"))
        
        self.grid_size_spin = QDoubleSpinBox()
        self.grid_size_spin.setRange(0.1, 100.0)
        self.grid_size_spin.setDecimals(1)
        self.grid_size_spin.setSingleStep(0.1)
        self.grid_size_spin.setValue(10.0)
        self.grid_size_spin.setSuffix(" %")
        self.grid_size_spin.setToolTip("Grid cell size as percentage of image size")
        self.grid_size_spin.valueChanged.connect(self.update_grid_size)
        right_sidebar_layout.addWidget(self.grid_size_spin)

        right_sidebar_layout.addSpacing(6)
        right_sidebar_layout.addWidget(QLabel("Grid Line Thickness:"))
        self.grid_thickness_spin = QSpinBox()
        self.grid_thickness_spin.setRange(1, 20)
        self.grid_thickness_spin.setValue(2)
        self.grid_thickness_spin.setSuffix(" px")
        self.grid_thickness_spin.setToolTip("Grid line thickness in pixels")
        self.grid_thickness_spin.valueChanged.connect(self.update_grid_line_thickness)
        right_sidebar_layout.addWidget(self.grid_thickness_spin)

        right_sidebar_layout.addSpacing(6)
        right_sidebar_layout.addWidget(QLabel("Polygon Line Thickness:"))
        self.polygon_thickness_spin = QSpinBox()
        self.polygon_thickness_spin.setRange(1, 20)
        self.polygon_thickness_spin.setValue(1)
        self.polygon_thickness_spin.setSuffix(" px")
        self.polygon_thickness_spin.setToolTip("Polygon line thickness in pixels")
        self.polygon_thickness_spin.valueChanged.connect(self.update_polygon_line_thickness)
        right_sidebar_layout.addWidget(self.polygon_thickness_spin)

        right_sidebar_layout.addSpacing(6)
        right_sidebar_layout.addWidget(QLabel("Scale Polygon Array:"))
        scale_poly_layout = QHBoxLayout()
        self.poly_scale_spin = QDoubleSpinBox()
        self.poly_scale_spin.setRange(0.1, 10000.0)
        self.poly_scale_spin.setDecimals(1)
        self.poly_scale_spin.setSingleStep(0.1)
        self.poly_scale_spin.setValue(100.0)
        self.poly_scale_spin.setSuffix(" %")
        self.poly_scale_spin.setToolTip("Scale all polygon coordinates by this percentage")
        scale_poly_layout.addWidget(self.poly_scale_spin)
        apply_scale_btn = QPushButton("Apply")
        apply_scale_btn.setToolTip("Scale all polygon coordinates")
        apply_scale_btn.clicked.connect(self.apply_polygon_scale)
        scale_poly_layout.addWidget(apply_scale_btn)
        right_sidebar_layout.addLayout(scale_poly_layout)

        right_sidebar_layout.addSpacing(10)
        right_sidebar_layout.addWidget(QLabel("Move Image:"))

        from PyQt5.QtWidgets import QCheckBox
        self.move_polygons_chk = QCheckBox("Move Polygons")
        self.move_polygons_chk.setToolTip("When checked, polygons and circles move together with the image")
        right_sidebar_layout.addWidget(self.move_polygons_chk)

        grid_arrows_layout = QGridLayout()
        grid_arrows_layout.setSpacing(2)

        up_btn = QPushButton("\u2191")
        up_btn.setFixedSize(36, 28)
        up_btn.clicked.connect(lambda: self.move_grid(0, -1))
        grid_arrows_layout.addWidget(up_btn, 0, 1)

        left_btn = QPushButton("\u2190")
        left_btn.setFixedSize(36, 28)
        left_btn.clicked.connect(lambda: self.move_grid(-1, 0))
        grid_arrows_layout.addWidget(left_btn, 1, 0)

        reset_grid_btn = QPushButton("\u25cb")
        reset_grid_btn.setFixedSize(36, 28)
        reset_grid_btn.setToolTip("Reset image position")
        reset_grid_btn.clicked.connect(self.reset_grid_offset)
        grid_arrows_layout.addWidget(reset_grid_btn, 1, 1)

        right_btn = QPushButton("\u2192")
        right_btn.setFixedSize(36, 28)
        right_btn.clicked.connect(lambda: self.move_grid(1, 0))
        grid_arrows_layout.addWidget(right_btn, 1, 2)

        down_btn = QPushButton("\u2193")
        down_btn.setFixedSize(36, 28)
        down_btn.clicked.connect(lambda: self.move_grid(0, 1))
        grid_arrows_layout.addWidget(down_btn, 2, 1)

        right_sidebar_layout.addLayout(grid_arrows_layout)

        right_sidebar_layout.addSpacing(5)
        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("Step (px):"))
        self.grid_step_spin = QSpinBox()
        self.grid_step_spin.setRange(1, 500)
        self.grid_step_spin.setValue(10)
        self.grid_step_spin.setToolTip("Pixels to move grid per arrow click")
        step_layout.addWidget(self.grid_step_spin)
        right_sidebar_layout.addLayout(step_layout)

        right_sidebar_layout.addSpacing(20)
        right_sidebar_layout.addWidget(QLabel("<b>Tile Export Settings</b>"))
        right_sidebar_layout.addSpacing(10)
        
        right_sidebar_layout.addWidget(QLabel("DPI:"))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 1200)
        self.dpi_spin.setValue(300)
        self.dpi_spin.setToolTip("Resolution in dots per inch")
        right_sidebar_layout.addWidget(self.dpi_spin)
        
        right_sidebar_layout.addSpacing(10)
        right_sidebar_layout.addWidget(QLabel("Tile Size:"))
        self.tile_size_spin = QSpinBox()
        self.tile_size_spin.setRange(10, 5000)
        self.tile_size_spin.setValue(200)
        self.tile_size_spin.setSuffix(" mm")
        self.tile_size_spin.setToolTip("Tile size in millimeters")
        right_sidebar_layout.addWidget(self.tile_size_spin)
        
        right_sidebar_layout.addSpacing(20)
        
        self.save_tile_btn = QPushButton("Save Tile Image")
        self.save_tile_btn.clicked.connect(self.save_tile_image)
        self.save_tile_btn.setToolTip("Save C3 tile as JPEG")
        right_sidebar_layout.addWidget(self.save_tile_btn)
        
        right_sidebar_layout.addStretch()

        # Add sidebar, scroll area, and right sidebar to main layout
        main_layout.addWidget(sidebar_widget)
        main_layout.addWidget(self.scroll_area)
        main_layout.addWidget(right_sidebar_widget)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
        # Connect signals
        self.canvas.selection_changed.connect(self.on_selection_changed)

    def on_selection_changed(self, index):
        if index == -1:
            self.effects_group.setEnabled(False)
        else:
            self.effects_group.setEnabled(True)
            # Load effects for this polygon
            effects = self.canvas.polygon_effects[index]
            
            # Block signals to prevent triggering update_effects loop
            self.brightness_slider.blockSignals(True)
            self.contrast_slider.blockSignals(True)
            self.saturation_slider.blockSignals(True)
            self.warmth_slider.blockSignals(True)
            self.black_point_slider.blockSignals(True)
            self.tint_strength_slider.blockSignals(True)
            
            self.brightness_slider.setValue(int(effects.get('brightness', 0)))
            self.contrast_slider.setValue(int(effects.get('contrast', 1.0) * 100))
            self.saturation_slider.setValue(int(effects.get('saturation', 1.0) * 100))
            self.warmth_slider.setValue(int(effects.get('warmth', 0)))
            self.black_point_slider.setValue(int(effects.get('black_point', 0)))
            self.tint_strength_slider.setValue(int(effects.get('tint_strength', 0)))
            
            # Update tint button color
            color = effects.get('tint_color', (255, 255, 255))
            self.tint_btn.setStyleSheet(f"background-color: rgb({color[0]}, {color[1]}, {color[2]}); color: black;")

            self.brightness_slider.blockSignals(False)
            self.contrast_slider.blockSignals(False)
            self.saturation_slider.blockSignals(False)
            self.warmth_slider.blockSignals(False)
            self.black_point_slider.blockSignals(False)
            self.tint_strength_slider.blockSignals(False)

    def select_tint_color(self):
        if self.canvas.selected_polygon_index is not None:
            idx = self.canvas.selected_polygon_index
            effects = self.canvas.polygon_effects[idx]
            current_color = effects.get('tint_color', (255, 255, 255))
            
            color = QColorDialog.getColor(QColor(current_color[0], current_color[1], current_color[2]), self, "Select Tint Color")
            
            if color.isValid():
                rgb = (color.red(), color.green(), color.blue())
                effects['tint_color'] = rgb
                self.tint_btn.setStyleSheet(f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); color: black;")
                self.canvas.apply_effects()

    def copy_colors(self):
        if self.canvas.selected_polygon_index is not None:
            idx = self.canvas.selected_polygon_index
            self.copied_effects = copy.deepcopy(self.canvas.polygon_effects[idx])
            self.paste_btn.setEnabled(True)
            QMessageBox.information(self, "Info", "Colors copied.")

    def paste_colors(self):
        if self.canvas.selected_polygon_index is not None and self.copied_effects:
            idx = self.canvas.selected_polygon_index
            self.canvas.polygon_effects[idx] = copy.deepcopy(self.copied_effects)
            
            # Update UI
            self.on_selection_changed(idx)
            
            # Apply effects
            self.canvas.apply_effects()
            QMessageBox.information(self, "Info", "Colors pasted.")

    def update_effects(self):
        if self.canvas.selected_polygon_index is not None:
            idx = self.canvas.selected_polygon_index
            effects = self.canvas.polygon_effects[idx]
            
            effects['brightness'] = self.brightness_slider.value()
            effects['contrast'] = self.contrast_slider.value() / 100.0
            effects['saturation'] = self.saturation_slider.value() / 100.0
            effects['warmth'] = self.warmth_slider.value()
            effects['black_point'] = self.black_point_slider.value()
            effects['tint_strength'] = self.tint_strength_slider.value()
            
            self.canvas.apply_effects()

    def _make_sharpness_slider(self):
        self.sharpness_slider = QSlider(Qt.Horizontal)
        self.sharpness_slider.setRange(0, 100)
        self.sharpness_slider.setValue(0)
        self.sharpness_slider.valueChanged.connect(self.update_global_sharpness)
        return self.sharpness_slider

    def update_global_sharpness(self):
        self.canvas.global_sharpness = self.sharpness_slider.value()
        self.canvas.apply_effects()

    def update_image_tilt(self):
        self.canvas.image_tilt = self.image_tilt_slider.value()
        self.canvas.apply_effects()
    
    def update_vertical_tilt(self):
        self.canvas.vertical_tilt = self.vertical_tilt_slider.value()
        self.canvas.apply_effects()

    def start_stretch_mode(self):
        # Uncheck polygon / circle buttons if checked
        if self.polygon_btn.isChecked():
            self.polygon_btn.setChecked(False)
            self.canvas.stop_polygon_drawing()
        if self.circle_btn.isChecked():
            self.circle_btn.setChecked(False)
            self.canvas.stop_circle_drawing()
        self.canvas.start_stretch_selection()

    def toggle_polygon_mode(self):
        if self.polygon_btn.isChecked():
            if self.circle_btn.isChecked():
                self.circle_btn.setChecked(False)
                self.canvas.stop_circle_drawing()
            success = self.canvas.start_polygon_drawing()
            if not success:
                self.polygon_btn.setChecked(False)
        else:
            self.canvas.stop_polygon_drawing()

    def toggle_circle_mode(self):
        if self.circle_btn.isChecked():
            if self.polygon_btn.isChecked():
                self.polygon_btn.setChecked(False)
                self.canvas.stop_polygon_drawing()
            success = self.canvas.start_circle_drawing()
            if not success:
                self.circle_btn.setChecked(False)
        else:
            self.canvas.stop_circle_drawing()
    
    def toggle_grid(self):
        self.canvas.show_grid = self.grid_btn.isChecked()
        self.canvas.update_canvas_size()
        self.canvas.update()

    def update_grid_size(self):
        self.canvas.grid_size_percent = self.grid_size_spin.value()
        self.canvas.update_canvas_size()
        self.canvas.update()

    def update_grid_line_thickness(self):
        self.canvas.grid_line_thickness = self.grid_thickness_spin.value()
        self.canvas.update()

    def update_polygon_line_thickness(self):
        self.canvas.polygon_line_thickness = self.polygon_thickness_spin.value()
        self.canvas.update()

    def apply_polygon_scale(self):
        if not self.canvas.polygons:
            QMessageBox.warning(self, "Warning", "No polygons to scale.")
            return
        # Snapshot originals on first scale so subsequent scales are always relative to original
        if not hasattr(self, '_original_polygons') or self._original_polygons is None:
            self._original_polygons = [list(poly) for poly in self.canvas.polygons]
        scale = self.poly_scale_spin.value() / 100.0
        self.canvas.polygons = [
            [(x * scale, y * scale) for x, y in poly]
            for poly in self._original_polygons
        ]
        self.canvas.apply_effects()
        self.canvas.update()

    def pan_image(self, dx, dy):
        step = self.grid_step_spin.value()
        hbar = self.scroll_area.horizontalScrollBar()
        vbar = self.scroll_area.verticalScrollBar()
        hbar.setValue(hbar.value() + dx * step)
        vbar.setValue(vbar.value() + dy * step)

    def reset_pan(self):
        self.scroll_area.horizontalScrollBar().setValue(0)
        self.scroll_area.verticalScrollBar().setValue(0)

    def move_grid(self, dx, dy):
        step = self.grid_step_spin.value()
        delta_x = dx * step
        delta_y = dy * step
        self.canvas.grid_offset_x += delta_x
        self.canvas.grid_offset_y += delta_y
        if self.move_polygons_chk.isChecked():
            for circ in self.canvas.circles:
                circ[0] += delta_x
                circ[1] += delta_y
            self.canvas.polygons = [
                [(px + delta_x, py + delta_y) for px, py in poly]
                for poly in self.canvas.polygons
            ]
        self.canvas.update()

    def reset_grid_offset(self):
        delta_x = -self.canvas.grid_offset_x
        delta_y = -self.canvas.grid_offset_y
        if self.move_polygons_chk.isChecked():
            for circ in self.canvas.circles:
                circ[0] += delta_x
                circ[1] += delta_y
            self.canvas.polygons = [
                [(px + delta_x, py + delta_y) for px, py in poly]
                for poly in self.canvas.polygons
            ]
        self.canvas.grid_offset_x = 0
        self.canvas.grid_offset_y = 0
        self.canvas.update()
    
    def save_tile_image(self):
        """Open a grid-selection popup then save chosen tiles as JPEG files."""
        if self.canvas.cv_image is None:
            QMessageBox.warning(self, "Warning", "No image loaded. Please load an image first.")
            return

        img_height, img_width = self.canvas.cv_image.shape[:2]
        smaller_dimension = min(img_width, img_height)
        grid_cell_size = int(smaller_dimension * (self.canvas.grid_size_percent / 100.0))

        if grid_cell_size == 0:
            QMessageBox.warning(self, "Warning", "Grid size is too small. Please increase grid size percentage.")
            return

        num_cols = max(1, int(np.ceil(img_width  / grid_cell_size)))
        num_rows = max(1, int(np.ceil(img_height / grid_cell_size)))

        # ── Grid-selection dialog ──────────────────────────────────────────
        from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QCheckBox,
                                     QScrollArea as _QSA, QWidget as _QW)

        dlg = QDialog(self)
        dlg.setWindowTitle("Select Grid Tiles to Save")
        dlg_layout = QVBoxLayout(dlg)

        info = QLabel(f"Grid: {num_rows} rows × {num_cols} columns  "
                      f"(cell size {grid_cell_size} px)")
        dlg_layout.addWidget(info)

        # Select-all / deselect-all buttons
        sel_btns = QHBoxLayout()
        sel_all_btn   = QPushButton("Select All")
        desel_all_btn = QPushButton("Deselect All")
        sel_btns.addWidget(sel_all_btn)
        sel_btns.addWidget(desel_all_btn)
        dlg_layout.addLayout(sel_btns)

        # Scrollable checkbox grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(2)
        scroll.setWidget(grid_widget)
        dlg_layout.addWidget(scroll)

        # Build checkboxes  { (row_idx, col_idx): QCheckBox }
        checkboxes = {}
        for r in range(num_rows):
            row_letter = chr(ord('A') + r) if r < 26 else f"R{r+1}"
            for c in range(num_cols):
                label = f"{row_letter}{c + 1}"
                cb = QCheckBox(label)
                grid_layout.addWidget(cb, r, c)
                checkboxes[(r, c)] = cb

        def select_all():
            for cb in checkboxes.values():
                cb.setChecked(True)

        def deselect_all():
            for cb in checkboxes.values():
                cb.setChecked(False)

        sel_all_btn.clicked.connect(select_all)
        desel_all_btn.clicked.connect(deselect_all)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)
        dlg_layout.addWidget(btn_box)

        dlg.resize(min(120 * num_cols + 40, 900),
                   min(40  * num_rows + 120, 700))

        if dlg.exec_() != QDialog.Accepted:
            return

        selected = [(r, c) for (r, c), cb in checkboxes.items() if cb.isChecked()]
        if not selected:
            QMessageBox.information(self, "Info", "No tiles selected.")
            return

        # ── Choose output folder ───────────────────────────────────────────
        from PyQt5.QtWidgets import QFileDialog as _QFD
        folder = QFileDialog.getExistingDirectory(self, "Choose Output Folder")
        if not folder:
            return

        # ── Export selected tiles ──────────────────────────────────────────
        tile_size_mm = self.tile_size_spin.value()
        dpi          = self.dpi_spin.value()
        target_pixels = int((tile_size_mm / 25.4) * dpi)
        source_image  = (self.canvas.display_image
                         if self.canvas.display_image is not None
                         else self.canvas.cv_image)

        import os
        saved, skipped = [], []
        for (row_index, col_index) in selected:
            row_letter = chr(ord('A') + row_index) if row_index < 26 else f"R{row_index+1}"
            tile_name  = f"{row_letter}{col_index + 1}"

            x_start = int(col_index * grid_cell_size - self.canvas.grid_offset_x)
            y_start = int(row_index * grid_cell_size - self.canvas.grid_offset_y)
            x_end   = x_start + grid_cell_size
            y_end   = y_start + grid_cell_size

            # Skip tiles fully outside image
            if x_start >= img_width or y_start >= img_height or x_end <= 0 or y_end <= 0:
                skipped.append(tile_name)
                continue

            x_start_c = max(0, x_start)
            y_start_c = max(0, y_start)
            x_end_c   = min(x_end, img_width)
            y_end_c   = min(y_end, img_height)

            tile_image  = source_image[y_start_c:y_end_c, x_start_c:x_end_c]
            tile_resized = cv2.resize(tile_image, (target_pixels, target_pixels),
                                      interpolation=cv2.INTER_LANCZOS4)
            tile_bgr = cv2.cvtColor(tile_resized, cv2.COLOR_RGB2BGR)

            out_path = os.path.join(folder, f"{tile_name}.jpg")
            if cv2.imwrite(out_path, tile_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                saved.append(tile_name)
            else:
                skipped.append(tile_name)

        msg = f"Saved {len(saved)} tile(s) to:\n{folder}"
        if skipped:
            msg += f"\n\nSkipped (out of bounds): {', '.join(skipped)}"
        QMessageBox.information(self, "Done", msg)

    def update_resolution(self):
        self.canvas.set_target_resolution(self.width_spin.value(), self.height_spin.value())

    def apply_image_scale(self):
        if self.canvas.cv_image is None:
            return
        h, w = self.canvas.cv_image.shape[:2]
        new_w = max(1, int(round(w * self.img_width_scale_spin.value() / 100.0)))
        new_h = max(1, int(round(h * self.img_height_scale_spin.value() / 100.0)))
        self.canvas.cv_image = cv2.resize(self.canvas.cv_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        self.width_spin.blockSignals(True)
        self.height_spin.blockSignals(True)
        self.width_spin.setValue(new_w)
        self.height_spin.setValue(new_h)
        self.width_spin.blockSignals(False)
        self.height_spin.blockSignals(False)
        self.canvas.set_target_resolution(new_w, new_h)
        self.img_width_scale_spin.setValue(100.0)
        self.img_height_scale_spin.setValue(100.0)
        self.canvas.apply_effects()
        self.canvas.update()

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            has_project = bool(self.canvas.polygons or self.canvas.circles)
            self.canvas.load_image(path)
            if self.canvas.cv_image is not None:
                h, w = self.canvas.cv_image.shape[:2]
                if not has_project:
                    # Fresh load — update spinboxes and auto-fit
                    self.width_spin.blockSignals(True)
                    self.height_spin.blockSignals(True)
                    self.width_spin.setValue(w)
                    self.height_spin.setValue(h)
                    self.width_spin.blockSignals(False)
                    self.height_spin.blockSignals(False)
                    self.canvas.set_target_resolution(w, h)
                    vp = self.scroll_area.viewport()
                    fit_scale = min(vp.width() / w, vp.height() / h)
                    self.canvas.scale_factor = fit_scale
                    self.canvas.update_canvas_size()
                    self.canvas.update()

    def save_image(self):
        if self.canvas.image is None:
            QMessageBox.warning(self, "Warning", "No image to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", "stretched.png", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.canvas.image.save(path)

    def save_circle_image(self):
        """Save the image region inside the selected circle at full original resolution."""
        if self.canvas.cv_image is None:
            QMessageBox.warning(self, "Warning", "No image loaded.")
            return
        if self.canvas.selected_circle_index is None or not self.canvas.circles:
            QMessageBox.warning(self, "Warning", "No circle selected. Click a circle to select it first.")
            return

        cx, cy, radius = self.canvas.circles[self.canvas.selected_circle_index]
        img_h, img_w = self.canvas.cv_image.shape[:2]

        # Bounding box of the circle
        x1 = int(cx - radius)
        y1 = int(cy - radius)
        x2 = int(cx + radius)
        y2 = int(cy + radius)

        # Clamp to image bounds
        x1c = max(0, x1)
        y1c = max(0, y1)
        x2c = min(img_w, x2)
        y2c = min(img_h, y2)

        if x2c <= x1c or y2c <= y1c:
            QMessageBox.warning(self, "Warning", "Circle is outside the image bounds.")
            return

        # Crop from the original unscaled image
        crop = self.canvas.cv_image[y1c:y2c, x1c:x2c].copy()

        # Circle centre relative to the (potentially clamped) crop
        local_cx = cx - x1
        local_cy = cy - y1
        # Offset by any clamping
        local_cx -= (x1c - x1)
        local_cy -= (y1c - y1)

        crop_h, crop_w = crop.shape[:2]
        Y, X = np.ogrid[:crop_h, :crop_w]
        mask = ((X - local_cx) ** 2 + (Y - local_cy) ** 2 <= radius ** 2).astype(np.uint8) * 255

        # Build RGBA output
        crop_rgba = cv2.cvtColor(crop, cv2.COLOR_RGB2RGBA)
        crop_rgba[:, :, 3] = mask

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Circle Image", "circle.png", "PNG Images (*.png)"
        )
        if path:
            cv2.imwrite(path, cv2.cvtColor(crop_rgba, cv2.COLOR_RGBA2BGRA))
            QMessageBox.information(self, "Success", f"Circle image saved to:\n{path}")
    
    def save_array(self):
        """Save polygons to CSV file compatible with mosaic_editor_pyqt"""
        if not self.canvas.polygons:
            QMessageBox.warning(self, "Warning", "No polygons to save.")
            return

        # Compute the current grid cell size in pixels
        original_cell_px = 1.0
        if self.canvas.cv_image is not None:
            img_h, img_w = self.canvas.cv_image.shape[:2]
            smaller = min(img_w, img_h)
            original_cell_px = max(1.0, smaller * (self.canvas.grid_size_percent / 100.0))

        from PyQt5.QtWidgets import QInputDialog
        new_cell_px, ok = QInputDialog.getDouble(
            self,
            "Calibrate Grid Box Size",
            f"Current grid box is {original_cell_px:.2f} px.\n"
            "Enter the target grid box size in pixels.\n"
            "All coordinates will be scaled by (target / current):",
            value=round(original_cell_px, 2),
            min=0.01,
            max=100000.0,
            decimals=2,
        )
        if not ok:
            return

        scale = new_cell_px / original_cell_px

        # Open file dialog to choose save location
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Array as CSV",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )

        if not filename:
            return  # User cancelled

        # Ask whether to save image colors or just polygon shapes
        reply = QMessageBox.question(
            self,
            "Save Colors?",
            "Do you want to save the mean image color for each polygon?\n\n"
            "Yes – fill each polygon with the average color from the image.\n"
            "No  – save polygons only (white fill).",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        save_colors = (reply == QMessageBox.Yes)

        # Grid origin is fixed at (0,0) in the canvas coordinate system.
        # Polygon points are already stored in that system, so no shift is needed.
        origin_x = 0.0
        origin_y = 0.0

        try:
            import csv
            import json
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                # Write header with frame color support and group ID
                writer.writerow(['polygon_id', 'coordinates', 'color_r', 'color_g', 'color_b', 'color_a',
                               'frame_r', 'frame_g', 'frame_b', 'frame_a', 'group_id'])

                # Write each polygon
                for i, points in enumerate(self.canvas.polygons):
                    # Translate so A1 corner is (0,0), then rescale
                    adjusted_points = [[(float(x) - origin_x) * scale,
                                        (float(y) - origin_y) * scale] for x, y in points]

                    coords_json = json.dumps(adjusted_points)

                    # Compute mean color of the image region covered by this polygon
                    r, g, b, a = 1.0, 1.0, 1.0, 1.0
                    if save_colors and self.canvas.display_image is not None:
                        img = self.canvas.display_image  # RGB uint8
                        h_img, w_img = img.shape[:2]
                        mask = np.zeros((h_img, w_img), dtype=np.uint8)
                        pts_int = np.array([[int(x), int(y)] for x, y in points], dtype=np.int32)
                        cv2.fillPoly(mask, [pts_int], 255)
                        if mask.any():
                            mean_rgb = img[mask == 255].mean(axis=0)  # [R, G, B]
                            r = float(mean_rgb[0]) / 255.0
                            g = float(mean_rgb[1]) / 255.0
                            b = float(mean_rgb[2]) / 255.0

                    fr, fg, fb, fa = 0.0, 0.0, 0.0, 1.0
                    group_id = ''

                    writer.writerow([i, coords_json, r, g, b, a, fr, fg, fb, fa, group_id])

            QMessageBox.information(
                self,
                "Success",
                f"Saved {len(self.canvas.polygons)} polygons to {filename}\n"
                f"Scale factor applied: {scale:.4f}  ({original_cell_px:.2f} px → {new_cell_px:.2f} px)"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save array: {str(e)}")
    
    def load_array(self):
        """Load polygons from CSV file compatible with mosaic_editor_pyqt"""
        # Open file dialog to choose file
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Array from CSV",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not filename:
            return  # User cancelled
        
        # Ask for scale factor
        from PyQt5.QtWidgets import QInputDialog
        scale_text, ok = QInputDialog.getText(
            self,
            "Scale Factor",
            "Enter scale percentage (default 100%):",
            text="100"
        )
        
        if not ok:
            return  # User cancelled
        
        try:
            scale_factor = float(scale_text) / 100.0  # Convert percentage to decimal
            scale_factor = max(0.01, min(1000.0, scale_factor))  # Clamp between 1% and 1000%
        except ValueError:
            scale_factor = 1.0  # Default to 100% if invalid input
        
        try:
            import csv
            import json
            polygons = []
            
            with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row_num, row in enumerate(reader, 1):
                    try:
                        # Check if this is the image parameters row
                        coords_str = row['coordinates'] if 'coordinates' in row else row.get('polygon_coords', '')
                        
                        # Skip rows with empty coordinates or special parameter rows
                        if not coords_str or coords_str.strip() == '' or coords_str in ['IMAGE_PARAMS', 'GRID_PARAMS']:
                            continue
                        
                        # Parse coordinates - handle JSON array format
                        coords_str = coords_str.strip('"\'')
                        
                        try:
                            coord_list = json.loads(coords_str)
                            # Apply scale factor
                            points = []
                            for point in coord_list:
                                scaled_x = float(point[0]) * scale_factor
                                scaled_y = float(point[1]) * scale_factor
                                points.append((scaled_x, scaled_y))
                        except:
                            # Fallback to ast parsing for backward compatibility
                            import ast
                            coord_list = ast.literal_eval(coords_str)
                            points = []
                            for point in coord_list:
                                scaled_x = float(point[0]) * scale_factor
                                scaled_y = float(point[1]) * scale_factor
                                points.append((scaled_x, scaled_y))
                        
                        if len(points) < 3:
                            continue
                        
                        polygons.append(points)
                        
                    except Exception as e:
                        print(f"Error parsing row {row_num}: {e}")
                        continue
            
            if polygons:
                # Clear existing polygons and load new ones
                self.canvas.polygons = polygons
                self._original_polygons = None  # reset scale snapshot
                # Also clear polygon effects and reset selection
                self.canvas.polygon_effects = [{
                    'brightness': 0,
                    'contrast': 1.0,
                    'saturation': 1.0,
                    'warmth': 0,
                    'tint_color': (255, 255, 255),
                    'tint_strength': 0,
                    'black_point': 0
                } for _ in polygons]
                self.canvas.selected_polygon_index = None
                self.canvas.dragging_point_index = None
                self.canvas.selection_changed.emit(-1)
                self.canvas.apply_effects()
                self.canvas.update()
                
                QMessageBox.information(
                    self, 
                    "Success", 
                    f"Loaded {len(polygons)} polygons from {filename} with {scale_factor*100:.1f}% scale"
                )
            else:
                QMessageBox.warning(self, "Warning", "No valid polygons found in the file.")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load array: {str(e)}")

    def save_project(self):
        if self.canvas.cv_image is None:
            QMessageBox.warning(self, "Warning", "No project to save.")
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "Save Project", "project.msp", "Mosaic Stretch Project (*.msp)")
        if path:
            data = {
                'image': self.canvas.cv_image,
                'polygons': self.canvas.polygons,
                'effects': self.canvas.polygon_effects,
                'width': self.canvas.target_width,
                'height': self.canvas.target_height,
                'image_tilt': self.canvas.image_tilt,
                'vertical_tilt': self.canvas.vertical_tilt,
                'show_grid': self.canvas.show_grid,
                'grid_size_percent': self.canvas.grid_size_percent,
                'grid_offset_x': self.canvas.grid_offset_x,
                'grid_offset_y': self.canvas.grid_offset_y,
                'circles': self.canvas.circles,
                'tile_size_mm': self.tile_size_spin.value(),
                'dpi': self.dpi_spin.value(),
            }
            try:
                with open(path, 'wb') as f:
                    pickle.dump(data, f)
                QMessageBox.information(self, "Success", "Project saved successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save project: {str(e)}")

    def load_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Project", "", "Mosaic Stretch Project (*.msp)")
        if path:
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                
                self.canvas.cv_image = data['image']
                self.canvas.polygons = data['polygons']
                self._original_polygons = None  # reset scale snapshot
                self.canvas.polygon_effects = data['effects']
                self.canvas.target_width = data.get('width', 300)
                self.canvas.target_height = data.get('height', 300)
                self.canvas.image_tilt = data.get('image_tilt', 0)
                self.canvas.vertical_tilt = data.get('vertical_tilt', 0)
                self.canvas.show_grid = data.get('show_grid', False)
                self.canvas.grid_size_percent = data.get('grid_size_percent', 10)
                self.canvas.grid_offset_x = data.get('grid_offset_x', 0)
                self.canvas.grid_offset_y = data.get('grid_offset_y', 0)
                self.canvas.circles = data.get('circles', [])
                
                # Update UI controls
                self.width_spin.setValue(self.canvas.target_width)
                self.height_spin.setValue(self.canvas.target_height)
                self.image_tilt_slider.setValue(self.canvas.image_tilt)
                self.vertical_tilt_slider.setValue(self.canvas.vertical_tilt)
                self.grid_btn.setChecked(self.canvas.show_grid)
                self.grid_size_spin.setValue(self.canvas.grid_size_percent)
                self.tile_size_spin.setValue(data.get('tile_size_mm', self.tile_size_spin.value()))
                self.dpi_spin.setValue(data.get('dpi', self.dpi_spin.value()))
                
                # Reset state
                self.canvas.display_image = self.canvas.cv_image.copy()
                self.canvas.scale_factor = 1.0
                self.canvas.points = []
                self.canvas.current_polygon = []
                self.canvas.selecting_mode = False
                self.canvas.drawing_polygon = False
                self.canvas.selected_polygon_index = None
                self.canvas.dragging_point_index = None
                
                # Refresh
                self.canvas.apply_effects() # This calls update_image_from_cv
                self.canvas.update()
                
                # Update effects panel state
                self.on_selection_changed(-1)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load project: {str(e)}")


    def eventFilter(self, source, event):
        if source == self.scroll_area.viewport() and event.type() == QEvent.Wheel:
            self.canvas.handle_zoom(event, event.pos())
            return True
        return super().eventFilter(source, event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())
