#!/usr/bin/env python3
"""
Mosaic Cutter - Simplified Polygon Viewer
A lightweight application for loading and viewing CSV polygon arrays with zoom functionality.
"""

import sys
import csv
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QFileDialog, QLabel, 
                           QMessageBox, QCheckBox, QLineEdit, QDialog)
from PyQt5.QtCore import Qt, QPointF, QRect
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QPolygonF, QFont, QWheelEvent
from shapely.geometry import Polygon
from shapely.ops import unary_union
import random


class CutterCanvas(QWidget):
    """Canvas widget for displaying polygons with zoom functionality"""
    
    def __init__(self):
        super().__init__()
        self.polygons = []
        self.colors = []
        self.edge_colors = []  # Store edge/frame colors separately
        self.original_colors = []  # Store original colors before Cut operation
        self.tile_polygons = {}  # Track tile polygons by box index {box_index: polygon}
        self.boxes_with_polygons = set()  # Track which grid boxes have polygons
        self.filled_box_index = -1  # Track which box should be filled (-1 = none)
        
        # View transformation parameters
        self.scale_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        
        # Bounds calculation
        self.min_x = 0
        self.max_x = 100
        self.min_y = 0
        self.max_y = 100
        
        # Mouse handling
        self.last_mouse_pos = None
        self.is_panning = False
        
        # Measure tool
        self.measure_mode = False
        self.measure_points = []  # Up to 2 world-coordinate tuples
        
        # Grid variables
        self.show_grid = False  # Whether to show the grid
        self.grid_size = 300  # Size of each individual grid box/cell in world coordinates
        self.grid_offset_x = 0  # Grid offset in world coordinates
        self.grid_offset_y = 0  # Grid offset in world coordinates
        self.grid_dragging = False  # Whether we're dragging the grid
        self.grid_drag_start = None  # Starting point for grid drag
        self.grid_drag_world_start = None  # Starting world coordinates for grid drag
        
        # Visual settings
        self.background_color = QColor(255, 255, 255)  # White background for canvas
        self.edge_color = QColor(0, 0, 0)
        self.edge_width = 1.0
        self.transparent_fill = False  # Whether to draw polygons with transparent fill
        
        self.setMinimumSize(800, 600)
        self.setFocusPolicy(Qt.StrongFocus)  # Allow keyboard focus for events
        
    def load_polygons_from_csv(self, filename):
        """Load polygons from CSV file (same logic as mosaic editor)"""
        try:
            polygons = []
            colors = []
            
            with open(filename, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row_num, row in enumerate(reader, 1):
                    try:
                        # Parse coordinates - handle JSON array format
                        coords_str = row['coordinates'] if 'coordinates' in row else row.get('polygon_coords', '')
                        
                        # Remove quotes and parse as JSON-like structure
                        coords_str = coords_str.strip('"\'')
                        
                        # Parse JSON-like coordinate array
                        try:
                            import ast
                            coord_list = ast.literal_eval(coords_str)
                            coords = [(float(point[0]), float(point[1])) for point in coord_list]
                        except:
                            # Fallback to manual parsing
                            coords_str = coords_str.strip('[]')
                            # Split by closing and opening brackets
                            coord_pairs = []
                            parts = coords_str.split('], [')
                            for part in parts:
                                part = part.strip('[]')
                                if ', ' in part:
                                    x_str, y_str = part.split(', ')
                                    try:
                                        x = float(x_str)
                                        y = float(y_str)
                                        coord_pairs.append((x, y))
                                    except ValueError:
                                        continue
                            coords = coord_pairs
                        
                        if len(coords) < 3:
                            continue
                        
                        # Create polygon
                        polygon = Polygon(coords)
                        if not polygon.is_valid:
                            continue
                        polygons.append(polygon)
                        
                        # Parse color - handle separate R,G,B columns or combined color column
                        if 'color_r' in row and 'color_g' in row and 'color_b' in row:
                            # Separate RGB columns (with optional alpha)
                            try:
                                r = float(row['color_r'])
                                g = float(row['color_g'])
                                b = float(row['color_b'])
                                
                                # Check for alpha channel (backward compatibility)
                                if 'color_a' in row:
                                    a = float(row['color_a'])
                                    a = int(a * 255) if a <= 1.0 else int(a)
                                else:
                                    a = 255  # Default to fully opaque if no alpha
                                
                                # Convert from 0-1 range to 0-255
                                r = int(r * 255) if r <= 1.0 else int(r)
                                g = int(g * 255) if g <= 1.0 else int(g)
                                b = int(b * 255) if b <= 1.0 else int(b)
                                
                                colors.append(QColor(r, g, b, a))
                            except ValueError as ve:
                                colors.append(QColor(128, 128, 128))  # Default gray
                        elif 'color' in row:
                            # Combined color column
                            color_str = row['color'].strip('()[]"\'')
                            
                            # Split color components
                            if ', ' in color_str:
                                color_parts = color_str.split(', ')
                            elif ',' in color_str:
                                color_parts = color_str.split(',')
                            else:
                                color_parts = color_str.split()
                            
                            if len(color_parts) >= 3:
                                try:
                                    r = float(color_parts[0].strip())
                                    g = float(color_parts[1].strip())
                                    b = float(color_parts[2].strip())
                                    
                                    # Handle both 0-1 and 0-255 ranges
                                    if r <= 1.0 and g <= 1.0 and b <= 1.0:
                                        r, g, b = int(r*255), int(g*255), int(b*255)
                                    else:
                                        r, g, b = int(r), int(g), int(b)
                                    
                                    colors.append(QColor(r, g, b))
                                except ValueError as ve:
                                    colors.append(QColor(128, 128, 128))  # Default gray
                            else:
                                colors.append(QColor(128, 128, 128))  # Default gray
                        else:
                            # No color information, use default
                            colors.append(QColor(128, 128, 128))
                            
                    except Exception as e:
                        continue
            
            if not polygons:
                print(f"No valid polygons found in {filename}")
                return False
                
            self.polygons = polygons
            self.colors = colors
            self.edge_colors = [QColor(0, 0, 0) for _ in polygons]  # Initialize edge colors to black
            self.original_colors = colors.copy()  # Save original colors before any modifications
            self.calculate_bounds()
            self.zoom_to_fit()
            self.update()
            print(f"Loaded {len(polygons)} polygons from {filename}")
            return True
            
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return False
    
    def generate_random_color(self):
        """Generate a random color for polygons without color data"""
        return QColor(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    
    def calculate_bounds(self):
        """Calculate the bounding box of all polygons"""
        if not self.polygons:
            return
            
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        
        for polygon in self.polygons:
            bounds = polygon.bounds
            min_x = min(min_x, bounds[0])
            min_y = min(min_y, bounds[1])
            max_x = max(max_x, bounds[2])
            max_y = max(max_y, bounds[3])
        
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
    
    def zoom_to_fit(self):
        """Zoom to fit all polygons in the view"""
        if not self.polygons:
            return
            
        # Calculate the scale factor to fit the content
        content_width = self.max_x - self.min_x
        content_height = self.max_y - self.min_y
        
        if content_width == 0 or content_height == 0:
            return
            
        # Add 10% margin
        margin = 0.1
        widget_width = self.width() * (1 - margin)
        widget_height = self.height() * (1 - margin)
        
        scale_x = widget_width / content_width
        scale_y = widget_height / content_height
        
        # Use the smaller scale to ensure everything fits
        self.scale_factor = min(scale_x, scale_y)
        
        # Center the content
        content_center_x = (self.min_x + self.max_x) / 2
        content_center_y = (self.min_y + self.max_y) / 2
        
        widget_center_x = self.width() / 2
        widget_center_y = self.height() / 2
        
        self.pan_x = widget_center_x - content_center_x * self.scale_factor
        self.pan_y = widget_center_y - content_center_y * self.scale_factor
        
        self.update()
    
    def world_to_screen(self, x, y):
        """Convert world coordinates to screen coordinates"""
        screen_x = x * self.scale_factor + self.pan_x
        screen_y = y * self.scale_factor + self.pan_y
        return screen_x, screen_y
    
    def screen_to_world(self, screen_x, screen_y):
        """Convert screen coordinates to world coordinates"""
        world_x = (screen_x - self.pan_x) / self.scale_factor
        world_y = (screen_y - self.pan_y) / self.scale_factor
        return world_x, world_y
    
    def paintEvent(self, event):
        """Paint the polygons"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        # Fill background
        painter.fillRect(self.rect(), self.background_color)
        
        if not self.polygons:
            # Draw instructions
            painter.setPen(QPen(QColor(100, 100, 100), 1))
            painter.setFont(QFont('Arial', 14))
            painter.drawText(self.rect(), Qt.AlignCenter, 
                           "Load a CSV file to view polygons\n\nControls:\n• Mouse wheel: Zoom\n• Left click + drag: Pan\n• F key: Zoom to Fit")
            return
        
        # Draw polygons
        for i, polygon in enumerate(self.polygons):
            color = self.colors[i] if i < len(self.colors) else QColor(100, 100, 100)
            edge_color = self.edge_colors[i] if i < len(self.edge_colors) else QColor(0, 0, 0)
            
            # Convert polygon coordinates to screen coordinates
            if hasattr(polygon, 'exterior'):
                # Single Polygon
                coords = list(polygon.exterior.coords)
                qt_polygon = QPolygonF()
                
                for x, y in coords:
                    screen_x, screen_y = self.world_to_screen(x, y)
                    qt_polygon.append(QPointF(screen_x, screen_y))
                
                # Draw polygon
                if self.transparent_fill:
                    painter.setBrush(QBrush(Qt.NoBrush))  # No fill, only outline
                else:
                    painter.setBrush(QBrush(color))  # Use original color fill
                painter.setPen(QPen(edge_color, 2))
                painter.drawPolygon(qt_polygon)
                
            elif hasattr(polygon, 'geoms'):
                # MultiPolygon - draw each polygon separately
                for sub_polygon in polygon.geoms:
                    if hasattr(sub_polygon, 'exterior'):
                        coords = list(sub_polygon.exterior.coords)
                        qt_polygon = QPolygonF()
                        
                        for x, y in coords:
                            screen_x, screen_y = self.world_to_screen(x, y)
                            qt_polygon.append(QPointF(screen_x, screen_y))
                        
                        # Draw polygon
                        if self.transparent_fill:
                            painter.setBrush(QBrush(Qt.NoBrush))  # No fill, only outline
                        else:
                            painter.setBrush(QBrush(color))  # Use original color fill
                        painter.setPen(QPen(edge_color, 1))
                        painter.drawPolygon(qt_polygon)
        
        # Draw black area polygon if it exists (from Cut Plates)
        if hasattr(self, 'black_area_main_polygon') and self.black_area_main_polygon:
            coords = list(self.black_area_main_polygon.exterior.coords)
            qt_polygon = QPolygonF()
            
            for x, y in coords:
                screen_x, screen_y = self.world_to_screen(x, y)
                qt_polygon.append(QPointF(screen_x, screen_y))
            
            # Draw with bright green outline for high visibility
            painter.setBrush(QBrush(Qt.NoBrush))
            painter.setPen(QPen(QColor(0, 255, 0), 1))  # Bright green, 1 pixel line
            painter.drawPolygon(qt_polygon)
            
            # Also draw center point as a marker
            center = self.black_area_main_polygon.centroid
            center_screen_x, center_screen_y = self.world_to_screen(center.x, center.y)
            painter.setBrush(QBrush(QColor(0, 255, 0)))
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            painter.drawEllipse(int(center_screen_x - 5), int(center_screen_y - 5), 10, 10)
        
        # Draw info text
        painter.setPen(QPen(QColor(50, 50, 50), 1))
        painter.setFont(QFont('Arial', 10))
        info_text = f"Polygons: {len(self.polygons)} | Zoom: {self.scale_factor:.2f}x"
        painter.drawText(10, self.height() - 10, info_text)
        
        # Draw measure points and line
        if self.measure_mode and self.measure_points:
            painter.setPen(QPen(QColor(255, 50, 50), 2))
            painter.setBrush(QBrush(QColor(255, 50, 50)))
            screen_pts = []
            for wx, wy in self.measure_points:
                sx, sy = self.world_to_screen(wx, wy)
                screen_pts.append((int(sx), int(sy)))
                painter.drawEllipse(int(sx) - 5, int(sy) - 5, 10, 10)
            if len(screen_pts) == 2:
                painter.drawLine(screen_pts[0][0], screen_pts[0][1],
                                 screen_pts[1][0], screen_pts[1][1])
        
        # Draw grid if enabled
        if self.show_grid:
            self.draw_grid(painter)
    
    def draw_grid(self, painter):
        """Draw the 6x6 grid overlay with draggable handle that scales with zoom"""
        
        # Calculate grid position and size in world coordinates
        # grid_size is now the size of each individual box/cell
        cell_size_world = self.grid_size
        total_grid_size_world = cell_size_world * 6  # 6 boxes = total grid size
        grid_x_world = self.grid_offset_x
        grid_y_world = self.grid_offset_y
        
        # Convert grid corners to screen coordinates
        grid_x_screen, grid_y_screen = self.world_to_screen(grid_x_world, grid_y_world)
        grid_end_x_screen, grid_end_y_screen = self.world_to_screen(
            grid_x_world + total_grid_size_world, grid_y_world + total_grid_size_world)
        
        # Calculate screen cell size
        cell_size_screen = (grid_end_x_screen - grid_x_screen) / 6
        
        # Draw grid lines
        painter.setPen(QPen(QColor(0, 0, 255), 2))  # Blue grid lines
        
        # Fill specified box if needed
        if self.filled_box_index >= 0:
            row = self.filled_box_index // 6
            col = self.filled_box_index % 6
            
            # Calculate box coordinates
            box_x_screen = grid_x_screen + col * cell_size_screen
            box_y_screen = grid_y_screen + row * cell_size_screen
            
            # Fill the box with black
            painter.fillRect(int(box_x_screen), int(box_y_screen), 
                           int(cell_size_screen), int(cell_size_screen), 
                           QColor(0, 0, 0))  # Black fill
        
        # Draw 6x6 grid (7 lines in each direction to create 6 boxes)
        for i in range(7):
            # Vertical lines
            x_screen = grid_x_screen + (i * cell_size_screen)
            painter.drawLine(int(x_screen), int(grid_y_screen), 
                           int(x_screen), int(grid_end_y_screen))
            
            # Horizontal lines
            y_screen = grid_y_screen + (i * cell_size_screen)
            painter.drawLine(int(grid_x_screen), int(y_screen), 
                           int(grid_end_x_screen), int(y_screen))
        
        # Draw column numbers (1-6) at the top of each column
        painter.setPen(QPen(QColor(0, 0, 255), 1))
        font = QFont()
        font.setPixelSize(max(12, int(cell_size_screen / 8)))  # Scale font with grid
        painter.setFont(font)
        
        for i in range(6):  # 6 columns
            column_center_x = grid_x_screen + (i + 0.5) * cell_size_screen
            number_y = grid_y_screen - 10  # Position above the grid
            painter.drawText(int(column_center_x - 5), int(number_y), str(i + 1))
        
        # Draw row letters (A-F) on the left side of each row
        for i in range(6):  # 6 rows
            row_center_y = grid_y_screen + (i + 0.5) * cell_size_screen
            letter_x = grid_x_screen - 20  # Position to the left of the grid
            painter.drawText(int(letter_x), int(row_center_y + 5), chr(ord('A') + i))
        
        # Draw draggable handle (small square in top-left corner)
        # Handle size scales with zoom but has min/max limits
        handle_size_world = max(10, min(50, 20 / self.scale_factor))
        handle_x_world = grid_x_world - handle_size_world / 2
        handle_y_world = grid_y_world - handle_size_world / 2
        
        handle_x_screen, handle_y_screen = self.world_to_screen(handle_x_world, handle_y_world)
        handle_end_x_screen, handle_end_y_screen = self.world_to_screen(
            handle_x_world + handle_size_world, handle_y_world + handle_size_world)
        
        handle_rect = QRect(int(handle_x_screen), int(handle_y_screen),
                           int(handle_end_x_screen - handle_x_screen),
                           int(handle_end_y_screen - handle_y_screen))
        
        # Fill handle with semi-transparent blue
        painter.fillRect(handle_rect, QColor(0, 0, 255, 128))
        
        # Draw handle border
        painter.setPen(QPen(QColor(0, 0, 255), 2))
        painter.drawRect(handle_rect)
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        # Get mouse position before zoom
        mouse_x = event.x()
        mouse_y = event.y()
        world_x, world_y = self.screen_to_world(mouse_x, mouse_y)
        
        # Calculate zoom factor
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        
        # Apply zoom
        new_scale = self.scale_factor * zoom_factor
        
        # Limit zoom range
        if 0.01 <= new_scale <= 100.0:
            self.scale_factor = new_scale
            
            # Adjust pan to keep mouse position fixed
            new_screen_x, new_screen_y = self.world_to_screen(world_x, world_y)
            self.pan_x += mouse_x - new_screen_x
            self.pan_y += mouse_y - new_screen_y
            
            self.update()
    
    def mousePressEvent(self, event):
        """Handle mouse press for panning and grid dragging"""
        if event.button() == Qt.LeftButton:
            # Measure mode: collect points and show distance
            if self.measure_mode:
                world_x, world_y = self.screen_to_world(event.x(), event.y())
                self.measure_points.append((world_x, world_y))
                if len(self.measure_points) == 2:
                    dx = self.measure_points[1][0] - self.measure_points[0][0]
                    dy = self.measure_points[1][1] - self.measure_points[0][1]
                    distance = (dx ** 2 + dy ** 2) ** 0.5
                    QMessageBox.information(
                        self, "Measurement",
                        f"Distance: {distance:.2f} units\n\n"
                        f"Point 1: ({self.measure_points[0][0]:.2f}, {self.measure_points[0][1]:.2f})\n"
                        f"Point 2: ({self.measure_points[1][0]:.2f}, {self.measure_points[1][1]:.2f})"
                    )
                    self.measure_points = []
                self.update()
                return  # Don't pan/drag while measuring
            
            # Check if we're clicking on the grid handle
            if self.show_grid and self.is_point_on_grid_handle(event.x(), event.y()):
                # Start grid dragging
                self.grid_dragging = True
                self.grid_drag_start = event.pos()
                self.grid_drag_world_start = (self.grid_offset_x, self.grid_offset_y)
            else:
                # Start panning
                self.last_mouse_pos = event.pos()
                self.is_panning = True
    
    def is_point_on_grid_handle(self, screen_x, screen_y):
        """Check if a screen point is on the grid handle"""
        if not self.show_grid:
            return False
            
        # Calculate handle position and size
        handle_size_world = max(10, min(50, 20 / self.scale_factor))
        handle_x_world = self.grid_offset_x - handle_size_world / 2
        handle_y_world = self.grid_offset_y - handle_size_world / 2
        
        handle_x_screen, handle_y_screen = self.world_to_screen(handle_x_world, handle_y_world)
        handle_end_x_screen, handle_end_y_screen = self.world_to_screen(
            handle_x_world + handle_size_world, handle_y_world + handle_size_world)
        
        # Check if point is within handle bounds
        return (handle_x_screen <= screen_x <= handle_end_x_screen and
                handle_y_screen <= screen_y <= handle_end_y_screen)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for panning and grid dragging"""
        if self.grid_dragging and self.grid_drag_start:
            # Handle grid dragging
            dx = event.x() - self.grid_drag_start.x()
            dy = event.y() - self.grid_drag_start.y()
            
            # Convert screen movement to world movement
            world_dx = dx / self.scale_factor
            world_dy = dy / self.scale_factor
            
            # Update grid position
            self.grid_offset_x = self.grid_drag_world_start[0] + world_dx
            self.grid_offset_y = self.grid_drag_world_start[1] + world_dy
            
            self.update()
        elif self.is_panning and self.last_mouse_pos:
            # Handle view panning
            dx = event.x() - self.last_mouse_pos.x()
            dy = event.y() - self.last_mouse_pos.y()
            
            self.pan_x += dx
            self.pan_y += dy
            
            self.last_mouse_pos = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        if event.button() == Qt.LeftButton:
            self.is_panning = False
            self.last_mouse_pos = None
            self.grid_dragging = False
            self.grid_drag_start = None
            self.grid_drag_world_start = None
    
    def keyPressEvent(self, event):
        """Handle keyboard events"""
        if event.key() == Qt.Key_F:
            # F key: Zoom to fit
            self.zoom_to_fit()
        elif event.key() == Qt.Key_R:
            # R key: Reset zoom
            self.scale_factor = 1.0
            self.pan_x = 0.0
            self.pan_y = 0.0
            self.update()
    
    def calculate_dominant_grid_box(self, polygon, grid_x, grid_y, cell_size):
        """Calculate which grid box contains most of the polygon's area"""
        from shapely.geometry import box
        
        # The polygon is already a Shapely polygon
        try:
            if not polygon.is_valid:
                return -1
        except:
            return -1
        
        max_overlap_area = 0
        best_box_index = -1
        
        # Check overlap with each of the 36 grid boxes (6x6)
        for row in range(6):
            for col in range(6):
                # Calculate box bounds
                box_x1 = grid_x + col * cell_size
                box_y1 = grid_y + row * cell_size
                box_x2 = box_x1 + cell_size
                box_y2 = box_y1 + cell_size
                
                # Create box geometry
                grid_box = box(box_x1, box_y1, box_x2, box_y2)
                
                try:
                    # Calculate intersection area
                    intersection = polygon.intersection(grid_box)
                    overlap_area = intersection.area
                    
                    if overlap_area > max_overlap_area:
                        max_overlap_area = overlap_area
                        best_box_index = row * 6 + col  # Convert 2D index to 1D
                except:
                    continue
        
        return best_box_index
    
    def invalidate_cache(self):
        """Invalidate any cached data (placeholder for compatibility with editor code)"""
        pass


class ControlPanel(QWidget):
    """Control panel for mosaic cutter settings"""
    
    def __init__(self, canvas=None):
        super().__init__()
        self.canvas = canvas
        self.frame_margin_percent = 10.0  # Default 10%
        self.offset_line_percent = 5.0    # Default 5%
        self.init_ui()
    
    def init_ui(self):
        """Initialize the control panel UI"""
        layout = QVBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Load CSV button
        self.load_btn = QPushButton("Load CSV")
        self.load_btn.clicked.connect(self.load_csv)
        layout.addWidget(self.load_btn)
        
        # Grid checkbox
        self.grid_checkbox = QCheckBox('Show Grid')
        self.grid_checkbox.setChecked(False)
        self.grid_checkbox.toggled.connect(self.on_grid_toggled)
        layout.addWidget(self.grid_checkbox)
        
        # Transparent checkbox
        self.transparent_checkbox = QCheckBox('Transparent Fill')
        self.transparent_checkbox.setChecked(False)
        self.transparent_checkbox.toggled.connect(self.on_transparent_toggled)
        layout.addWidget(self.transparent_checkbox)
        
        # Grid size input
        grid_size_label = QLabel('Box Size:')
        layout.addWidget(grid_size_label)
        
        self.grid_size_input = QLineEdit()
        self.grid_size_input.setText('300')
        self.grid_size_input.setPlaceholderText('Enter box size (e.g.,300)')
        self.grid_size_input.textChanged.connect(self.on_grid_size_changed)
        layout.addWidget(self.grid_size_input)
        
        # Frame margin percentage input
        frame_margin_label = QLabel('Frame Margin %:')
        layout.addWidget(frame_margin_label)
        
        self.frame_margin_input = QLineEdit()
        self.frame_margin_input.setText('10')
        self.frame_margin_input.setPlaceholderText('Enter margin % (e.g., 10)')
        self.frame_margin_input.textChanged.connect(self.on_frame_margin_changed)
        layout.addWidget(self.frame_margin_input)
        
        # Offset line percentage input
        offset_line_label = QLabel('Offset Line %:')
        layout.addWidget(offset_line_label)
        
        self.offset_line_input = QLineEdit()
        self.offset_line_input.setText('5')
        self.offset_line_input.setPlaceholderText('Enter offset % (e.g., 5)')
        self.offset_line_input.textChanged.connect(self.on_offset_line_changed)
        layout.addWidget(self.offset_line_input)
        
        # Cut button
        self.cut_btn = QPushButton("Cut")
        self.cut_btn.setEnabled(False)  # Initially disabled until CSV is loaded
        self.cut_btn.clicked.connect(self.on_cut_clicked)
        layout.addWidget(self.cut_btn)
        
        # Tiles button
        self.tiles_btn = QPushButton("Tiles")
        self.tiles_btn.setEnabled(False)  # Initially disabled until Cut is performed
        self.tiles_btn.clicked.connect(self.on_tiles_clicked)
        layout.addWidget(self.tiles_btn)
        
        # Save Boxes button
        self.save_boxes_btn = QPushButton("Save Boxes")
        self.save_boxes_btn.setEnabled(False)  # Initially disabled until Cut is performed
        self.save_boxes_btn.clicked.connect(self.on_save_boxes_clicked)
        layout.addWidget(self.save_boxes_btn)
        
        # Measure button
        self.measure_btn = QPushButton("Measure")
        self.measure_btn.setCheckable(True)
        self.measure_btn.clicked.connect(self.on_measure_clicked)
        layout.addWidget(self.measure_btn)
        
        # Add stretch to push everything to the top
        layout.addStretch()
        
        self.setLayout(layout)
        self.setFixedWidth(200)
        
        # Set the panel background color to be clearly different from canvas
        self.setStyleSheet("ControlPanel { background-color: #e0e0e0; border-left: 1px solid #c0c0c0; }")
    
    def load_csv(self):
        """Open file dialog and load CSV"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load CSV Polygon File",
            "",
            "CSV files (*.csv);;All files (*.*)"
        )
        
        if file_path:
            success = self.canvas.load_polygons_from_csv(file_path)
            if success:
                # Enable Cut button after successful CSV load
                self.cut_btn.setEnabled(True)
            else:
                QMessageBox.warning(self, "Error", "Failed to load CSV file. Please check the file format.")
    
    def on_grid_toggled(self, show_grid):
        """Handle grid checkbox toggle"""
        if self.canvas:
            self.canvas.show_grid = show_grid
            self.canvas.update()
    
    def on_transparent_toggled(self, transparent_fill):
        """Handle transparent fill checkbox toggle"""
        if self.canvas:
            self.canvas.transparent_fill = transparent_fill
            self.canvas.update()
    
    def on_grid_size_changed(self):
        """Handle grid size changes"""
        try:
            grid_size = int(self.grid_size_input.text()) if self.grid_size_input.text() else 300
            grid_size = max(10, grid_size)  # Ensure minimum grid size of 10
            if self.canvas:
                self.canvas.grid_size = grid_size
                self.canvas.update()
        except ValueError:
            # Invalid input, ignore
            pass
    
    def on_frame_margin_changed(self):
        """Handle frame margin percentage changes"""
        try:
            margin_percent = float(self.frame_margin_input.text()) if self.frame_margin_input.text() else 10.0
            margin_percent = max(0, min(margin_percent, 100))  # Clamp between 0 and 100
            self.frame_margin_percent = margin_percent
        except ValueError:
            # Invalid input, ignore
            pass
    
    def on_offset_line_changed(self):
        """Handle offset line percentage changes"""
        try:
            offset_percent = float(self.offset_line_input.text()) if self.offset_line_input.text() else 5.0
            offset_percent = max(0, min(offset_percent, 100))  # Clamp between 0 and 100
            self.offset_line_percent = offset_percent
        except ValueError:
            # Invalid input, ignore
            pass
    
    def on_measure_clicked(self, checked):
        """Handle Measure button toggle"""
        if self.canvas:
            self.canvas.measure_mode = checked
            self.canvas.measure_points = []
            self.canvas.setCursor(Qt.CrossCursor if checked else Qt.ArrowCursor)
            self.canvas.update()
    
    def on_cut_clicked(self):
        """Handle Cut button click - assign colors based on grid boxes"""
        if not self.canvas or not self.canvas.polygons:
            return
        
        # Disable Cut button to prevent multiple simultaneous executions
        self.cut_btn.setEnabled(False)
        self.cut_btn.setText("Processing...")
        # Force UI update to show the disabled state immediately
        QApplication.processEvents()
        
        # Get grid parameters
        cell_size = self.canvas.grid_size
        grid_x = self.canvas.grid_offset_x
        grid_y = self.canvas.grid_offset_y
        
        # Define 36 rainbow colors with maximum distinction and better contrast
        grid_colors = [
            QColor(255, 0, 0),     # Bright Red
            QColor(0, 255, 0),     # Bright Green  
            QColor(0, 0, 255),     # Bright Blue
            QColor(255, 255, 0),   # Bright Yellow
            QColor(255, 0, 255),   # Bright Magenta
            QColor(0, 255, 255),   # Bright Cyan
            QColor(255, 128, 0),   # Orange
            QColor(128, 0, 255),   # Purple
            QColor(0, 128, 0),     # Forest Green
            QColor(128, 0, 0),     # Maroon
            QColor(0, 0, 128),     # Navy Blue
            QColor(128, 128, 0),   # Olive
            QColor(255, 192, 203), # Pink
            QColor(165, 42, 42),   # Brown
            QColor(255, 165, 0),   # Dark Orange
            QColor(75, 0, 130),    # Indigo
            QColor(240, 230, 140), # Khaki
            QColor(220, 20, 60),   # Crimson
            QColor(32, 178, 170),  # Light Sea Green
            QColor(255, 20, 147),  # Deep Pink
            QColor(0, 191, 255),   # Deep Sky Blue
            QColor(154, 205, 50),  # Yellow Green
            QColor(255, 69, 0),    # Red Orange
            QColor(138, 43, 226),  # Blue Violet
            QColor(50, 205, 50),   # Lime Green
            QColor(255, 140, 0),   # Dark Orange 2
            QColor(72, 61, 139),   # Dark Slate Blue
            QColor(255, 215, 0),   # Gold
            QColor(199, 21, 133),  # Medium Violet Red
            QColor(0, 255, 127),   # Spring Green
            QColor(255, 105, 180), # Hot Pink
            QColor(30, 144, 255),  # Dodger Blue
            QColor(124, 252, 0),   # Lawn Green
            QColor(255, 0, 127),   # Rose
            QColor(64, 224, 208),  # Turquoise
            QColor(218, 165, 32)   # Goldenrod
        ]
        
        # Track which boxes contain polygons
        boxes_with_polygons = set()
        
        # Process each polygon and assign color based on dominant grid box
        new_colors = []
        for i, polygon in enumerate(self.canvas.polygons):
            # Calculate the dominant grid box for this polygon
            box_index = self.canvas.calculate_dominant_grid_box(polygon, grid_x, grid_y, cell_size)
            
            if box_index >= 0 and box_index < 36:
                # Use the color corresponding to the grid box
                color = grid_colors[box_index]
                boxes_with_polygons.add(box_index)
            else:
                # Fallback color for polygons that don't fit in any box
                color = QColor(128, 128, 128)  # Gray
            
            new_colors.append(color)
        
        # Update canvas with new colors
        self.canvas.colors = new_colors
        self.canvas.boxes_with_polygons = boxes_with_polygons
        
        # Debug: Print some color information
        print(f"Applied {len(new_colors)} colors. First few colors:")
        for i in range(min(5, len(new_colors))):
            color = new_colors[i]
            print(f"  Polygon {i}: RGB({color.red()}, {color.green()}, {color.blue()})")
        
        # Update the display
        self.canvas.invalidate_cache()
        self.canvas.update()
        
        # Re-enable Cut button after operation is complete
        self.cut_btn.setEnabled(True)
        self.cut_btn.setText("Cut")
        # Enable Save Boxes and Tiles buttons after Cut operation
        self.save_boxes_btn.setEnabled(True)
        self.tiles_btn.setEnabled(True)
        # Force UI update to show the re-enabled state
        QApplication.processEvents()
    
    def on_tiles_clicked(self):
        """Handle Tiles button click - create polygons for all grid boxes with content"""
        if not self.canvas or not self.canvas.polygons:
            return
            
        # Check if Cut has been run (we need the box assignments)
        if not hasattr(self.canvas, 'boxes_with_polygons'):
            QMessageBox.warning(self, "Error", "Please run Cut first to assign polygons to grid boxes.")
            return
        
        # Disable Tiles button during processing
        self.tiles_btn.setEnabled(False)
        self.tiles_btn.setText("Processing...")
        # Force UI update to show the disabled state immediately
        QApplication.processEvents()
        
        # Get grid parameters
        cell_size = self.canvas.grid_size
        grid_x = self.canvas.grid_offset_x
        grid_y = self.canvas.grid_offset_y
        
        # Initialize canvas arrays if needed
        if not hasattr(self.canvas, 'polygons'):
            self.canvas.polygons = []
            self.canvas.colors = []
            self.canvas.edge_colors = []
        
        # Count for summary
        total_boxes_processed = 0
        total_unified = 0
        total_subtracted = 0
        
        # Process each box that has polygons
        for box_index in self.canvas.boxes_with_polygons:
            # Convert 1D box index to 2D coordinates (row, col)
            row = box_index // 6
            col = box_index % 6
            
            # Calculate box bounds in world coordinates
            box_x1 = grid_x + col * cell_size
            box_y1 = grid_y + row * cell_size
            box_x2 = box_x1 + cell_size
            box_y2 = box_y1 + cell_size
            
            # Create a rectangular polygon for this box
            box_polygon_coords = [
                (box_x1, box_y1),  # Top-left
                (box_x2, box_y1),  # Top-right
                (box_x2, box_y2),  # Bottom-right
                (box_x1, box_y2),  # Bottom-left
                (box_x1, box_y1)   # Close the polygon
            ]
            
            # Create Shapely polygon for this box
            box_polygon = Polygon(box_polygon_coords)
            
            # Find polygons for this specific box
            box_assigned_polygons = []
            intersecting_other_polygons = []
            
            for i, polygon in enumerate(self.canvas.polygons):
                # Get the box assignment for this polygon from Cut results
                polygon_box_index = self.canvas.calculate_dominant_grid_box(polygon, grid_x, grid_y, cell_size)
                
                # Check if polygon intersects or touches this box boundary
                if polygon.intersects(box_polygon) or polygon.touches(box_polygon):
                    if polygon_box_index == box_index:
                        # This polygon was assigned to this box by Cut function
                        box_assigned_polygons.append(polygon)
                    else:
                        # This polygon intersects this box but was assigned to another box
                        intersecting_other_polygons.append(polygon)
            
            # Start with the original box polygon
            modified_box_polygon = box_polygon
            
            # First, unify all polygons that were assigned to this box by Cut
            for polygon_to_unify in box_assigned_polygons:
                try:
                    modified_box_polygon = modified_box_polygon.union(polygon_to_unify)
                    total_unified += 1
                except Exception as unify_e:
                    print(f"Error unifying polygon in box {box_index}: {unify_e}")
                    continue
            
            # Then subtract polygons that intersect this box but are assigned to other boxes
            for polygon_to_subtract in intersecting_other_polygons:
                try:
                    modified_box_polygon = modified_box_polygon.difference(polygon_to_subtract)
                    total_subtracted += 1
                except Exception as subtract_e:
                    print(f"Error subtracting polygon from box {box_index}: {subtract_e}")
                    continue
            
            # Add the final box polygon to the canvas
            self.canvas.polygons.append(modified_box_polygon)
            
            # Track this as a tile polygon
            if not hasattr(self.canvas, 'tile_polygons'):
                self.canvas.tile_polygons = {}
            self.canvas.tile_polygons[box_index] = modified_box_polygon
            
            # Use transparent fill or no fill based on user setting
            if hasattr(self.canvas, 'transparent_fill') and self.canvas.transparent_fill:
                self.canvas.colors.append(QColor(0, 0, 0, 0))  # Transparent
            else:
                # Use different colors for different boxes (cycling through colors)
                colors = [
                    QColor(255, 0, 0, 100),    # Red
                    QColor(0, 255, 0, 100),    # Green
                    QColor(0, 0, 255, 100),    # Blue
                    QColor(255, 255, 0, 100),  # Yellow
                    QColor(255, 0, 255, 100),  # Magenta
                    QColor(0, 255, 255, 100),  # Cyan
                ]
                color_index = box_index % len(colors)
                self.canvas.colors.append(colors[color_index])
            
            # All tiles get yellow outline
            self.canvas.edge_colors.append(QColor(255, 255, 0))   # Yellow outline
            
            total_boxes_processed += 1
            
            # Convert box index to letter+number format for logging
            box_letter = chr(ord('A') + row)
            box_number = col + 1
            print(f"Created tile polygon for box {box_letter}{box_number} (index {box_index})")
            print(f"  - Unified {len(box_assigned_polygons)} assigned polygons")
            print(f"  - Subtracted {len(intersecting_other_polygons)} intersecting polygons")
        
        # Update the display
        self.canvas.update()
        
        # Summary
        print(f"\n=== Tiles Creation Summary ===")
        print(f"Processed {total_boxes_processed} boxes with polygons")
        print(f"Total polygons unified: {total_unified}")
        print(f"Total polygons subtracted: {total_subtracted}")
        
        # Re-enable Tiles button after operation is complete
        self.tiles_btn.setEnabled(True)
        self.tiles_btn.setText("Tiles")
        # Force UI update to show the re-enabled state
        QApplication.processEvents()
    
    def on_save_boxes_clicked(self):
        """Handle Save Boxes button click - save polygons organized by grid boxes"""
        if not self.canvas or not self.canvas.polygons or not hasattr(self.canvas, 'boxes_with_polygons'):
            QMessageBox.warning(self, "Error", "No cut data available. Please run Cut first.")
            return
        
        # Open directory selection dialog
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Directory to Save Box Files",
            "",  # Start from current directory
            QFileDialog.ShowDirsOnly
        )
        
        if not output_dir:
            return  # User cancelled dialog
        
        # Ask user about fillet options
        fillet_dialog = QDialog(self)
        fillet_dialog.setWindowTitle("Fillet Options")
        fillet_dialog.setModal(True)
        fillet_dialog.resize(300, 150)
        
        layout = QVBoxLayout(fillet_dialog)
        
        # Checkbox for enabling fillet
        fillet_checkbox = QCheckBox("Apply fillet to polygon corners")
        layout.addWidget(fillet_checkbox)
        
        # Radius input
        radius_layout = QHBoxLayout()
        radius_label = QLabel("Fillet radius:")
        radius_input = QLineEdit("2.0")  # Default 2.0 units
        radius_input.setEnabled(False)  # Initially disabled
        radius_layout.addWidget(radius_label)
        radius_layout.addWidget(radius_input)
        layout.addLayout(radius_layout)
        
        # Enable/disable radius input based on checkbox
        def on_checkbox_changed(checked):
            radius_input.setEnabled(checked)
        fillet_checkbox.stateChanged.connect(on_checkbox_changed)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        buttons_layout.addWidget(ok_button)
        buttons_layout.addWidget(cancel_button)
        layout.addLayout(buttons_layout)
        
        # Button connections
        ok_button.clicked.connect(fillet_dialog.accept)
        cancel_button.clicked.connect(fillet_dialog.reject)
        
        # Show dialog and get result
        if fillet_dialog.exec_() != QDialog.Accepted:
            return  # User cancelled
        
        # Get fillet settings
        apply_fillet = fillet_checkbox.isChecked()
        fillet_radius = 0.0
        if apply_fillet:
            try:
                fillet_radius = float(radius_input.text())
                if fillet_radius <= 0:
                    QMessageBox.warning(self, "Invalid Input", "Fillet radius must be greater than 0.")
                    return
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Please enter a valid number for fillet radius.")
                return
        
        # Disable Save Boxes button during processing
        self.save_boxes_btn.setEnabled(False)
        self.save_boxes_btn.setText("Saving...")
        QApplication.processEvents()
        
        try:
            # Create output directory
            import os
            from datetime import datetime
            
            # Use the selected directory directly (no "box tiles" subdirectory)
            box_tiles_dir = output_dir
            
            # Get grid parameters
            cell_size = self.canvas.grid_size
            grid_x = self.canvas.grid_offset_x
            grid_y = self.canvas.grid_offset_y
            
            # Define box labels (A1-F6 for 6x6 grid)
            box_labels = []
            for row in range(6):
                for col in range(6):
                    row_letter = chr(ord('A') + row)  # A, B, C, D, E, F
                    col_number = col + 1  # 1, 2, 3, 4, 5, 6
                    box_labels.append(f"{row_letter}{col_number}")
            
            # Group polygons by box
            boxes_data = {}
            for box_index in self.canvas.boxes_with_polygons:
                if 0 <= box_index < 36:
                    box_label = box_labels[box_index]
                    boxes_data[box_label] = {
                        'polygons': [],
                        'colors': [],
                        'original_colors': [],  # Store original colors separately
                        'box_index': box_index
                    }
            
            # Assign polygons to boxes (excluding tile polygons)
            tile_polygons_set = set()
            if hasattr(self.canvas, 'tile_polygons'):
                tile_polygons_set = set(self.canvas.tile_polygons.values())
            
            for i, polygon in enumerate(self.canvas.polygons):
                # Skip tile polygons - they will be saved separately
                if polygon in tile_polygons_set:
                    continue
                    
                box_index = self.canvas.calculate_dominant_grid_box(polygon, grid_x, grid_y, cell_size)
                if 0 <= box_index < 36:
                    box_label = box_labels[box_index]
                    if box_label in boxes_data:
                        boxes_data[box_label]['polygons'].append(polygon)
                        boxes_data[box_label]['colors'].append(self.canvas.colors[i])
                        
                        # Get original color for this polygon
                        original_color = self.canvas.colors[i]  # Default to current color
                        if (hasattr(self.canvas, 'original_colors') and 
                            i < len(self.canvas.original_colors)):
                            original_color = self.canvas.original_colors[i]
                        
                        boxes_data[box_label]['original_colors'].append(original_color)
                        
                        # Store the original polygon index to retrieve original color later
                        if 'original_indices' not in boxes_data[box_label]:
                            boxes_data[box_label]['original_indices'] = []
                        boxes_data[box_label]['original_indices'].append(i)
            
            # Save each box as a separate CSV file
            saved_files = []
            dxf_files_saved = 0
            for box_label, data in boxes_data.items():
                if data['polygons']:
                    # Calculate box top-left coordinates for coordinate transformation
                    box_index = data['box_index']
                    row = box_index // 6
                    col = box_index % 6
                    box_offset_x = grid_x + col * cell_size
                    box_offset_y = grid_y + row * cell_size
                    
                    # Create box directory directly in selected folder
                    box_dir = os.path.join(box_tiles_dir, box_label)
                    if not os.path.exists(box_dir):
                        os.makedirs(box_dir)
                    
                    # Apply fillet to polygons if requested
                    processed_polygons = []
                    for polygon in data['polygons']:
                        if apply_fillet and fillet_radius > 0:
                            polygon = self.apply_fillet_to_polygon(polygon, fillet_radius)
                        processed_polygons.append(polygon)
                    
                    # Save box CSV file (with filleted polygons, ORIGINAL colors, and transformed coordinates)
                    csv_filename = os.path.join(box_dir, f"box_{box_label}.csv")
                    self.save_box_csv(csv_filename, processed_polygons, data['original_colors'], box_offset_x, box_offset_y)
                    saved_files.append(csv_filename)
                    
                    # Prepare polygon data for DXF export (using filleted polygons and ORIGINAL colors)
                    polygons_data = []
                    for i, polygon in enumerate(processed_polygons):
                        # Use original color for both display and DXF files
                        original_color = data['colors'][i]  # Default to current Cut color
                        if i < len(data['original_colors']):
                            original_color = data['original_colors'][i]  # Use stored original color
                        
                        polygons_data.append({
                            'polygon': polygon,
                            'color': original_color,  # Use original color
                            'original_color': original_color,  # Same for DXF
                            'original_index': i
                        })
                    
                    # Save main DXF file
                    dxf_filename = os.path.join(box_dir, f"box_{box_label}.dxf")
                    self.save_polygons_to_dxf(polygons_data, dxf_filename, box_label, data['box_index'])
                    dxf_files_saved += 1
                    
                    # Group polygons by original color for color-specific DXF files
                    color_groups = {}
                    for poly_data in polygons_data:
                        original_color = poly_data.get('original_color', poly_data['color'])
                        color_hex = original_color.name()  # Get hex color like #FF0000
                        
                        if color_hex not in color_groups:
                            color_groups[color_hex] = []
                        
                        # Use original color for the DXF file
                        poly_data_copy = poly_data.copy()
                        poly_data_copy['color'] = original_color
                        color_groups[color_hex].append(poly_data_copy)
                    
                    # Save separate DXF file for each color
                    for color_hex, color_polygons in color_groups.items():
                        color_dxf_filename = os.path.join(box_dir, f"{box_label}_{color_hex}.dxf")
                        try:
                            self.save_polygons_to_dxf(color_polygons, color_dxf_filename, f"{box_label} - {color_hex}", data['box_index'])
                            dxf_files_saved += 1
                        except Exception as e:
                            print(f"Failed to save color DXF {color_dxf_filename}: {str(e)}")
                    
                    # Save tile polygon DXF if it exists for this box
                    if hasattr(self.canvas, 'tile_polygons') and data['box_index'] in self.canvas.tile_polygons:
                        tile_polygon = self.canvas.tile_polygons[data['box_index']]
                        tile_dxf_filename = os.path.join(box_dir, f"{box_label}_tile.dxf")
                        
                        # Create polygon data for the tile polygon
                        tile_polygon_data = [{
                            'polygon': tile_polygon,
                            'color': QColor(255, 255, 0),  # Yellow color for tile polygon
                            'original_color': QColor(255, 255, 0),
                            'original_index': 0
                        }]
                        
                        try:
                            self.save_polygons_to_dxf(tile_polygon_data, tile_dxf_filename, f"{box_label} Tile", data['box_index'])
                            dxf_files_saved += 1
                            print(f"Saved tile polygon DXF: {tile_dxf_filename}")
                        except Exception as e:
                            print(f"Failed to save tile DXF {tile_dxf_filename}: {str(e)}")
            
            # Create general CSV file with all polygons and area calculations
            self.save_general_csv(box_tiles_dir, boxes_data)
            
            # Show success message
            QMessageBox.information(self, "Success", 
                                  f"Saved {len(saved_files)} box CSV files and {dxf_files_saved} DXF files to:\n{box_tiles_dir}\n\n"
                                  f"Boxes saved: {', '.join([os.path.basename(os.path.dirname(f)) for f in saved_files])}")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save boxes: {str(e)}")
        
        finally:
            # Re-enable Save Boxes button
            self.save_boxes_btn.setEnabled(True)
            self.save_boxes_btn.setText("Save Boxes")
            QApplication.processEvents()
    
    def get_black_area_polygon(self):
        """Extract polygon by combining all A1 polygons PLUS the A1 box boundary into one big polygon"""
        try:
            print("Starting polygon combination approach for A1 box...")
            
            # Get grid parameters
            cell_size = self.canvas.grid_size
            grid_x = self.canvas.grid_offset_x
            grid_y = self.canvas.grid_offset_y
            
            print(f"Grid parameters: size={cell_size}, offset=({grid_x}, {grid_y})")
            
            # Target A1 box (box_index = 0)
            target_box_index = 0
            
            # Create the A1 box polygon (rectangular boundary)
            a1_box_polygon = Polygon([
                (grid_x, grid_y),
                (grid_x + cell_size, grid_y), 
                (grid_x + cell_size, grid_y + cell_size),
                (grid_x, grid_y + cell_size),
                (grid_x, grid_y)  # Close the polygon
            ])
            
            print(f"Created A1 box polygon with area: {a1_box_polygon.area:.2f}")
            
            # Collect all polygons that belong to A1 box
            a1_polygons = [a1_box_polygon]  # Start with the box itself
            
            for i, polygon in enumerate(self.canvas.polygons):
                box_index = self.canvas.calculate_dominant_grid_box(polygon, grid_x, grid_y, cell_size)
                if box_index == target_box_index:
                    if polygon.is_valid and polygon.area > 0:
                        a1_polygons.append(polygon)
            
            print(f"Found {len(a1_polygons)-1} valid mosaic polygons in A1 box (plus the A1 box itself)")
            
            if len(a1_polygons) == 1:  # Only the box, no mosaic polygons
                print("Only A1 box found, no mosaic polygons")
                return a1_box_polygon
            
            # Print some stats about the polygons
            mosaic_area = sum(p.area for p in a1_polygons[1:])  # Exclude the box polygon
            print(f"A1 box area: {a1_box_polygon.area:.2f}")
            print(f"Total mosaic area in A1: {mosaic_area:.2f}")
            
            # Combine all polygons (A1 box + mosaic polygons) into one big polygon using unary_union
            print("Performing unary_union operation...")
            combined_polygon = unary_union(a1_polygons)
            
            # Handle the result (could be Polygon or MultiPolygon)
            if hasattr(combined_polygon, 'geoms'):
                # MultiPolygon result - get the largest part
                print(f"Result is MultiPolygon with {len(combined_polygon.geoms)} parts")
                largest = max(combined_polygon.geoms, key=lambda p: p.area)
                print(f"Using largest part with {len(list(largest.exterior.coords))} vertices and area {largest.area:.2f}")
                return largest
            else:
                # Single Polygon result
                if hasattr(combined_polygon, 'exterior'):
                    print(f"Result is single Polygon with {len(list(combined_polygon.exterior.coords))} vertices and area {combined_polygon.area:.2f}")
                    return combined_polygon
                else:
                    print(f"Unexpected result type: {type(combined_polygon)}")
                    return None
        
        except Exception as e:
            print(f"Error combining A1 polygons: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_general_csv(self, output_dir, boxes_data):
        """Save general CSV file with all polygons and color area summary"""
        try:
            import json
            import os
            
            general_csv_path = os.path.join(output_dir, "all_polygons_general.csv")
            color_areas = {}  # Dictionary to store total area for each color
            
            with open(general_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow(['polygon_id', 'box_name', 'coordinates', 'color_r', 'color_g', 'color_b', 'color_a', 'color_hex', 'area'])
                
                # Write all polygons from all boxes
                global_polygon_id = 0
                for box_label, data in boxes_data.items():
                    for i, polygon in enumerate(data['polygons']):
                        # Use original color for CSV files (before Cut operation)
                        original_color = data['colors'][i]  # Default to current color
                        if i < len(data['original_colors']):
                            original_color = data['original_colors'][i]  # Use stored original color
                        
                        # Calculate area
                        poly_area = polygon.area
                        
                        # Track color areas using original colors
                        color_key = original_color.name()  # Get hex color like #FF0000
                        if color_key not in color_areas:
                            color_areas[color_key] = 0
                        color_areas[color_key] += poly_area
                        
                        # Extract coordinates as a list of [x, y] pairs
                        coords = list(polygon.exterior.coords)
                        coords_json = json.dumps([[float(x), float(y)] for x, y in coords])
                        
                        # Extract RGBA values (convert from QColor to 0-1 range) - use original color
                        r = original_color.red() / 255.0
                        g = original_color.green() / 255.0
                        b = original_color.blue() / 255.0
                        a = original_color.alpha() / 255.0
                        
                        # Get hex color code
                        color_hex = original_color.name()  # Returns hex format like #FF0000
                        
                        # Write row
                        writer.writerow([global_polygon_id, box_label, coords_json, r, g, b, a, color_hex, poly_area])
                        global_polygon_id += 1
            
            # Save color summary CSV
            color_summary_path = os.path.join(output_dir, "color_area_summary.csv")
            with open(color_summary_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['color_hex', 'total_area', 'percentage'])
                
                # Calculate total area
                total_area = sum(color_areas.values())
                
                # Sort colors by area (largest first)
                sorted_colors = sorted(color_areas.items(), key=lambda x: x[1], reverse=True)
                
                for color_hex, area in sorted_colors:
                    percentage = (area / total_area * 100) if total_area > 0 else 0
                    writer.writerow([color_hex, area, f"{percentage:.2f}%"])
                    
        except Exception as e:
            print(f"Failed to save general CSV files: {str(e)}")
    
    def apply_fillet_to_polygon(self, polygon, radius):
        """Apply fillet (rounded corners) to a polygon using buffer operations"""
        try:
            if radius <= 0:
                return polygon
            
            # Apply negative buffer then positive buffer to create fillet effect
            # This rounds both inner and outer corners
            filleted = polygon.buffer(-radius).buffer(radius)
            
            # If the result is empty or invalid, return original
            if filleted.is_empty or not filleted.is_valid:
                return polygon
            
            return filleted
        except Exception as e:
            print(f"Error applying fillet to polygon: {e}")
            return polygon
    
    def save_box_csv(self, filename, polygons, colors, offset_x=0, offset_y=0):
        """Save polygons and colors to a CSV file with coordinate transformation"""
        import csv
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['coordinates', 'color_r', 'color_g', 'color_b', 'color_a', 'color_hex']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for polygon, color in zip(polygons, colors):
                # Handle both Polygon and MultiPolygon types
                polygons_to_save = []
                if hasattr(polygon, 'exterior'):
                    # Single Polygon
                    polygons_to_save.append(polygon)
                elif hasattr(polygon, 'geoms'):
                    # MultiPolygon - save each polygon separately
                    polygons_to_save.extend(polygon.geoms)
                else:
                    print(f"Warning: Unknown polygon type {type(polygon)}, skipping")
                    continue
                
                # Save each polygon (single or part of MultiPolygon)
                for sub_poly in polygons_to_save:
                    if not hasattr(sub_poly, 'exterior'):
                        continue  # Skip invalid geometries
                        
                    # Get coordinates and apply transformation
                    coords = list(sub_poly.exterior.coords[:-1])  # Remove duplicate last point
                    
                    # Transform coordinates by subtracting the box offset
                    transformed_coords = []
                    for x, y in coords:
                        new_x = x - offset_x
                        new_y = y - offset_y
                        transformed_coords.append((new_x, new_y))
                    
                    coord_str = str(transformed_coords)
                    
                    # Normalize color values to 0-1 range
                    r = color.red() / 255.0
                    g = color.green() / 255.0
                    b = color.blue() / 255.0
                    a = color.alpha() / 255.0
                    
                    # Get hex color code
                    color_hex = color.name()  # Returns hex format like #FF0000
                    
                    writer.writerow({
                        'coordinates': coord_str,
                        'color_r': r,
                        'color_g': g,
                        'color_b': b,
                        'color_a': a,
                        'color_hex': color_hex
                    })
    
    def save_polygons_to_dxf(self, polygons_data, dxf_filepath, box_name, box_index=None):
        """Save polygons to DXF file format using ezdxf library"""
        try:
            import ezdxf
            
            # Create a new DXF document
            doc = ezdxf.new('R2010')  # Use AutoCAD 2010 format for good compatibility
            msp = doc.modelspace()
            
            # Add title text
            msp.add_text(f"Grid Box {box_name}", dxfattribs={'height': 10, 'insert': (0, 0)})
            
            # Calculate frame based on grid box dimensions + 20 pixel margin
            if box_index is not None:
                # Get grid parameters
                cell_size_world = self.canvas.grid_size
                grid_x_world = self.canvas.grid_offset_x
                grid_y_world = self.canvas.grid_offset_y
                
                # Calculate grid box position
                row = box_index // 6
                col = box_index % 6
                
                # Calculate original grid box bounds
                box_min_x = grid_x_world + col * cell_size_world
                box_min_y = grid_y_world + row * cell_size_world
                box_max_x = box_min_x + cell_size_world
                box_max_y = box_min_y + cell_size_world
                
                # Use frame margin percentage from control panel
                frame_margin = cell_size_world * (self.frame_margin_percent / 100.0)
                frame_min_x = box_min_x - frame_margin
                frame_min_y = box_min_y - frame_margin
                frame_max_x = box_max_x + frame_margin
                frame_max_y = box_max_y + frame_margin
                
            elif polygons_data:
                # Fallback: calculate from polygons if box_index not provided
                min_x = float('inf')
                min_y = float('inf')
                max_x = float('-inf')
                max_y = float('-inf')
                
                # Find bounds of all polygons
                for poly_data in polygons_data:
                    polygon = poly_data['polygon']
                    bounds = polygon.bounds  # (minx, miny, maxx, maxy)
                    min_x = min(min_x, bounds[0])
                    min_y = min(min_y, bounds[1])
                    max_x = max(max_x, bounds[2])
                    max_y = max(max_y, bounds[3])
                
                # Add 20 pixels margin on each side for the frame
                frame_margin = 20
                frame_min_x = min_x - frame_margin
                frame_min_y = min_y - frame_margin
                frame_max_x = max_x + frame_margin
                frame_max_y = max_y + frame_margin
            else:
                frame_min_x = frame_min_y = frame_max_x = frame_max_y = None
            
            # Draw frame as a rectangle if we have valid frame dimensions
            if frame_min_x is not None:
                frame_points = [
                    (frame_min_x, frame_min_y),
                    (frame_max_x, frame_min_y),
                    (frame_max_x, frame_max_y),
                    (frame_min_x, frame_max_y),
                    (frame_min_x, frame_min_y)  # Close the rectangle
                ]
                msp.add_lwpolyline(frame_points, dxfattribs={'color': 8})  # Dark gray frame
            
            # Add 5% offset boundary for tile DXF or box DXF
            if len(polygons_data) > 0:
                # Calculate offset distance based on cell size using offset line percentage
                offset_distance = cell_size_world * (self.offset_line_percent / 100.0) if box_index is not None else 10
                
                try:
                    # Collect all polygons
                    all_polygons = []
                    for poly_data in polygons_data:
                        polygon = poly_data['polygon']
                        if hasattr(polygon, 'exterior'):
                            all_polygons.append(polygon)
                        elif hasattr(polygon, 'geoms'):
                            all_polygons.extend(polygon.geoms)
                    
                    # Union all polygons into one shape
                    if len(all_polygons) > 0:
                        unified_polygon = unary_union(all_polygons)
                        
                        # Create buffered (offset) polygon
                        buffered_polygon = unified_polygon.buffer(offset_distance, join_style=2)  # join_style=2 for mitered joins
                        
                        # Add the offset boundary as a blue line
                        if hasattr(buffered_polygon, 'exterior'):
                            offset_coords = list(buffered_polygon.exterior.coords)
                            if len(offset_coords) > 1 and offset_coords[0] == offset_coords[-1]:
                                offset_coords = offset_coords[:-1]
                            msp.add_lwpolyline(offset_coords, close=True, dxfattribs={'color': 5})  # Blue color
                        elif hasattr(buffered_polygon, 'geoms'):
                            # Handle MultiPolygon result
                            for geom in buffered_polygon.geoms:
                                if hasattr(geom, 'exterior'):
                                    offset_coords = list(geom.exterior.coords)
                                    if len(offset_coords) > 1 and offset_coords[0] == offset_coords[-1]:
                                        offset_coords = offset_coords[:-1]
                                    msp.add_lwpolyline(offset_coords, close=True, dxfattribs={'color': 5})
                except Exception as e:
                    print(f"Warning: Could not create offset boundary: {e}")
            
            # Add each polygon to the DXF
            for poly_id, poly_data in enumerate(polygons_data):
                polygon = poly_data['polygon']
                color = poly_data['color']
                
                # Use original color for DXF files (before Cut operation)
                original_color = color  # Default to current color
                if 'original_color' in poly_data:
                    original_color = poly_data['original_color']
                elif hasattr(self.canvas, 'original_colors') and len(self.canvas.original_colors) > 0:
                    # Find the original index of this polygon in the full list
                    try:
                        original_index = self.canvas.polygons.index(polygon)
                        if original_index < len(self.canvas.original_colors):
                            original_color = self.canvas.original_colors[original_index]
                    except (ValueError, IndexError):
                        pass  # Use current color as fallback
                
                # Handle both Polygon and MultiPolygon types
                polygons_to_process = []
                if hasattr(polygon, 'exterior'):
                    # Single Polygon
                    polygons_to_process.append(polygon)
                elif hasattr(polygon, 'geoms'):
                    # MultiPolygon - process each polygon separately
                    polygons_to_process.extend(polygon.geoms)
                else:
                    print(f"Warning: Unknown polygon type {type(polygon)}, skipping")
                    continue
                
                # Map ORIGINAL color to AutoCAD color index (not the Cut color)
                color_index = self.get_autocad_color_index(original_color)
                
                # Process each polygon (single polygon or each part of MultiPolygon)
                for sub_poly in polygons_to_process:
                    if not hasattr(sub_poly, 'exterior'):
                        continue  # Skip invalid geometries
                        
                    # Get coordinates and convert to the format ezdxf expects
                    coords = list(sub_poly.exterior.coords)
                    if len(coords) > 0:
                        # Remove duplicate closing point if it exists
                        if len(coords) > 1 and coords[0] == coords[-1]:
                            coords = coords[:-1]
                        
                        # Add the polygon as an LWPOLYLINE using ORIGINAL color
                        msp.add_lwpolyline(coords, close=True, dxfattribs={'color': color_index})
            
            # Save the DXF file
            doc.saveas(dxf_filepath)
            print(f"Successfully saved DXF file: {dxf_filepath}")
            
        except ImportError:
            print("Warning: ezdxf library not available, falling back to manual DXF creation")
            self.save_polygons_to_dxf_manual(polygons_data, dxf_filepath, box_name, box_index)
        except Exception as e:
            print(f"Error saving DXF file {dxf_filepath}: {e}")
            # Try fallback method
            try:
                self.save_polygons_to_dxf_manual(polygons_data, dxf_filepath, box_name, box_index)
            except Exception as e2:
                print(f"Fallback DXF creation also failed: {e2}")
    
    def save_polygons_to_dxf_manual(self, polygons_data, dxf_filepath, box_name, box_index=None):
        """Fallback manual DXF creation with proper structure"""
        with open(dxf_filepath, 'w', encoding='utf-8') as f:
            # Write proper DXF header with required sections
            f.write("0\nSECTION\n2\nHEADER\n")
            f.write("9\n$ACADVER\n1\nAC1015\n")  # AutoCAD 2000 format
            f.write("9\n$HANDSEED\n5\n20000\n")  # Handle seed
            f.write("0\nENDSEC\n")
            
            # Tables section with proper structure
            f.write("0\nSECTION\n2\nTABLES\n")
            f.write("0\nTABLE\n2\nLAYER\n5\n2\n330\n0\n100\nAcDbSymbolTable\n70\n1\n")
            f.write("0\nLAYER\n5\n10\n330\n2\n100\nAcDbSymbolTableRecord\n")
            f.write("100\nAcDbLayerTableRecord\n2\n0\n70\n0\n62\n7\n6\nCONTINUOUS\n")
            f.write("0\nENDTAB\n")
            f.write("0\nENDSEC\n")
            
            # Objects section (required for modern DXF)
            f.write("0\nSECTION\n2\nOBJECTS\n")
            f.write("0\nDICTIONARY\n5\nC\n330\n0\n100\nAcDbDictionary\n")
            f.write("0\nENDSEC\n")
            
            # Entities section
            f.write("0\nSECTION\n2\nENTITIES\n")
            
            handle_counter = 100  # Start handle counter
            
            # Add title text with proper structure
            f.write(f"0\nTEXT\n5\n{handle_counter:X}\n330\n1F\n100\nAcDbEntity\n")
            f.write(f"8\n0\n100\nAcDbText\n10\n0.0\n20\n0.0\n30\n0.0\n")
            f.write(f"40\n10.0\n1\nGrid Box {box_name}\n")
            handle_counter += 1
            
            # Calculate and add frame
            frame_points = self.calculate_frame_coordinates(box_index, polygons_data)
            if frame_points:
                f.write(f"0\nLWPOLYLINE\n5\n{handle_counter:X}\n330\n1F\n")
                f.write("100\nAcDbEntity\n8\n0\n62\n8\n")
                f.write("100\nAcDbPolyline\n90\n4\n70\n1\n")
                
                for x, y in frame_points[:4]:  # Only first 4 points for rectangle
                    f.write(f"10\n{x:.6f}\n20\n{y:.6f}\n")
                handle_counter += 1
            
            # Add 5% offset boundary for tile DXF or box DXF
            if len(polygons_data) > 0:
                # Calculate offset distance using offset line percentage
                if box_index is not None:
                    cell_size_world = self.canvas.grid_size
                    offset_distance = cell_size_world * (self.offset_line_percent / 100.0)
                else:
                    offset_distance = 10
                
                try:
                    # Collect all polygons
                    all_polygons = []
                    for poly_data in polygons_data:
                        polygon = poly_data['polygon']
                        if hasattr(polygon, 'exterior'):
                            all_polygons.append(polygon)
                        elif hasattr(polygon, 'geoms'):
                            all_polygons.extend(polygon.geoms)
                    
                    # Union all polygons into one shape
                    if len(all_polygons) > 0:
                        unified_polygon = unary_union(all_polygons)
                        
                        # Create buffered (offset) polygon
                        buffered_polygon = unified_polygon.buffer(offset_distance, join_style=2)
                        
                        # Handle both Polygon and MultiPolygon results
                        polygons_to_draw = []
                        if hasattr(buffered_polygon, 'exterior'):
                            polygons_to_draw.append(buffered_polygon)
                        elif hasattr(buffered_polygon, 'geoms'):
                            polygons_to_draw.extend(buffered_polygon.geoms)
                        
                        for geom in polygons_to_draw:
                            if hasattr(geom, 'exterior'):
                                offset_coords = list(geom.exterior.coords)
                                if len(offset_coords) > 1 and offset_coords[0] == offset_coords[-1]:
                                    offset_coords = offset_coords[:-1]
                                
                                if len(offset_coords) >= 3:
                                    f.write(f"0\nLWPOLYLINE\n5\n{handle_counter:X}\n330\n1F\n")
                                    f.write("100\nAcDbEntity\n8\n0\n62\n5\n")  # Color 5 = blue
                                    f.write("100\nAcDbPolyline\n")
                                    f.write(f"90\n{len(offset_coords)}\n70\n1\n")
                                    
                                    for x, y in offset_coords:
                                        f.write(f"10\n{x:.6f}\n20\n{y:.6f}\n")
                                    handle_counter += 1
                except Exception as e:
                    print(f"Warning: Could not create offset boundary: {e}")
            
            # Add polygons with proper LWPOLYLINE structure
            for poly_data in polygons_data:
                polygon = poly_data['polygon']
                color = poly_data['color']
                
                # Use original color for DXF files (before Cut operation)
                original_color = color  # Default to current color
                if 'original_color' in poly_data:
                    original_color = poly_data['original_color']
                elif hasattr(self.canvas, 'original_colors') and len(self.canvas.original_colors) > 0:
                    # Find the original index of this polygon in the full list
                    try:
                        original_index = self.canvas.polygons.index(polygon)
                        if original_index < len(self.canvas.original_colors):
                            original_color = self.canvas.original_colors[original_index]
                    except (ValueError, IndexError):
                        pass  # Use current color as fallback
                
                color_index = self.get_autocad_color_index(original_color)
                
                polygons_to_process = []
                if hasattr(polygon, 'exterior'):
                    polygons_to_process.append(polygon)
                elif hasattr(polygon, 'geoms'):
                    polygons_to_process.extend(polygon.geoms)
                
                for sub_poly in polygons_to_process:
                    if not hasattr(sub_poly, 'exterior'):
                        continue
                        
                    coords = list(sub_poly.exterior.coords)
                    if len(coords) > 1 and coords[0] == coords[-1]:
                        coords = coords[:-1]  # Remove duplicate closing point
                    
                    if len(coords) >= 3:  # Need at least 3 points for a polygon
                        f.write(f"0\nLWPOLYLINE\n5\n{handle_counter:X}\n330\n1F\n")
                        f.write("100\nAcDbEntity\n8\n0\n")
                        f.write(f"62\n{color_index}\n")
                        f.write("100\nAcDbPolyline\n")
                        f.write(f"90\n{len(coords)}\n70\n1\n")
                        
                        for x, y in coords:
                            f.write(f"10\n{x:.6f}\n20\n{y:.6f}\n")
                        handle_counter += 1
            
            f.write("0\nENDSEC\n")
            f.write("0\nEOF\n")
    
    def calculate_frame_coordinates(self, box_index, polygons_data):
        """Calculate frame coordinates for manual DXF creation"""
        if box_index is not None:
            cell_size_world = self.canvas.grid_size
            grid_x_world = self.canvas.grid_offset_x
            grid_y_world = self.canvas.grid_offset_y
            
            row = box_index // 6
            col = box_index % 6
            
            box_min_x = grid_x_world + col * cell_size_world
            box_min_y = grid_y_world + row * cell_size_world
            box_max_x = box_min_x + cell_size_world
            box_max_y = box_min_y + cell_size_world
            
            frame_margin = cell_size_world * (self.frame_margin_percent / 100.0)
            return [
                (box_min_x - frame_margin, box_min_y - frame_margin),
                (box_max_x + frame_margin, box_min_y - frame_margin),
                (box_max_x + frame_margin, box_max_y + frame_margin),
                (box_min_x - frame_margin, box_max_y + frame_margin),
                (box_min_x - frame_margin, box_min_y - frame_margin)
            ]
        elif polygons_data:
            min_x = min_y = float('inf')
            max_x = max_y = float('-inf')
            
            for poly_data in polygons_data:
                bounds = poly_data['polygon'].bounds
                min_x = min(min_x, bounds[0])
                min_y = min(min_y, bounds[1])
                max_x = max(max_x, bounds[2])
                max_y = max(max_y, bounds[3])
            
            # Calculate margin as 10% of average dimension
            avg_dimension = ((max_x - min_x) + (max_y - min_y)) / 2
            frame_margin = avg_dimension * 0.1
            return [
                (min_x - frame_margin, min_y - frame_margin),
                (max_x + frame_margin, min_y - frame_margin),
                (max_x + frame_margin, max_y + frame_margin),
                (min_x - frame_margin, max_y + frame_margin),
                (min_x - frame_margin, min_y - frame_margin)
            ]
        return None
    
    def get_autocad_color_index(self, color):
        """Convert QColor to AutoCAD color index (simplified mapping)"""
        # Basic color mapping to AutoCAD standard colors
        rgb = (color.red(), color.green(), color.blue())
        
        # Common color mappings
        color_map = {
            (255, 0, 0): 1,    # Red
            (255, 255, 0): 2,  # Yellow
            (0, 255, 0): 3,    # Green
            (0, 255, 255): 4,  # Cyan
            (0, 0, 255): 5,    # Blue
            (255, 0, 255): 6,  # Magenta
            (255, 255, 255): 7, # White
            (0, 0, 0): 0,      # Black
        }
        
        # Return mapped color or default to color by layer
        return color_map.get(rgb, 256)  # 256 = ByLayer


class MosaicCutter(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mosaic Cutter - Polygon Viewer")
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create canvas
        self.canvas = CutterCanvas()
        
        # Create control panel
        self.control_panel = ControlPanel(canvas=self.canvas)
        
        # Add widgets to main layout
        main_layout.addWidget(self.canvas, 1)  # Canvas takes most space
        main_layout.addWidget(self.control_panel, 0)  # Panel has fixed width
        
        # Set window state to maximized after UI is created
        self.setWindowState(Qt.WindowMaximized)
        
    def load_csv(self):
        """Load CSV through control panel"""
        self.control_panel.load_csv()
    
    def zoom_to_fit(self):
        """Zoom to fit through control panel"""
        self.control_panel.zoom_to_fit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Create and show the main window
    window = MosaicCutter()
    window.show()
    
    sys.exit(app.exec_())
