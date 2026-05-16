#!/usr/bin/env python3
"""
Polygon Duplicator Application
A simple PyQt5 application with a central canvas and side panels for polygon editing.
"""

import sys
import csv
import json
import math
import random
import numpy as np
from scipy.interpolate import splprep, splev
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QFrame, QLabel, QPushButton, QFileDialog, QCheckBox, QSpinBox, QLineEdit, QInputDialog, QMessageBox,
    QSizePolicy, QSlider, QColorDialog
)
from PyQt5.QtCore import Qt, QPoint, QTimer, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QPen, QPixmap, QBrush, QFont, QPolygon, QCursor, QLinearGradient

class ContinuousButton(QPushButton):
    """Custom button that supports continuous action when held down"""
    def __init__(self, text, callback, *args):
        super().__init__(text)
        self.callback = callback
        self.args = args
        self.press_timer = QTimer()
        self.repeat_timer = QTimer()
        
        # Set up timers
        self.press_timer.setSingleShot(True)
        self.press_timer.timeout.connect(self.start_continuous)
        
        self.repeat_timer.timeout.connect(self.execute_action)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Execute once immediately
            self.execute_action()
            # Start timer for continuous action after 1 second
            self.press_timer.start(1000)  # 1 second delay
        super().mousePressEvent(event)
        
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Stop all timers
            self.press_timer.stop()
            self.repeat_timer.stop()
        super().mouseReleaseEvent(event)
        
    def start_continuous(self):
        """Start continuous execution"""
        self.repeat_timer.start(50)  # Repeat every 50ms for smooth movement
        
    def execute_action(self):
        """Execute the callback with arguments"""
        self.callback(*self.args)


class Canvas(QWidget):
    """Central canvas widget for drawing/displaying content"""
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(600, 600)
        self.setStyleSheet("background-color: white; border: 1px solid black;")
        self.background_image = None
        self.original_background_image = None  # Store original image for scaling
        self.current_x_scale = 1.0  # Current X scale factor
        self.current_y_scale = 1.0  # Current Y scale factor
        
        # Polygon drawing mode variables
        self.polygon_mode = False
        self.polygon_points = []  # Points for the current polygon being drawn
        self.polygon_cursor_size = 10  # Size of the square cursor in pixels
        self.polygons = []  # List of completed polygons
        
        # Undo system
        self.undo_stack = []  # Stack of previous polygon states
        self.max_undo_states = 50  # Maximum number of undo states to keep
        
        # Zoom and pan variables
        self.zoom_factor = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        self.is_panning = False
        self.last_pan_point = None
        
        # Image offset variables (separate from zoom/pan)
        self.image_offset_x = 0
        self.image_offset_y = 0
        
        # Eraser mode
        self.eraser_mode = False
        self.is_erasing = False  # Track if currently dragging to erase
        
        # Duplicate mode
        self.duplicate_mode = False  # Whether to create 8 copies of each polygon
        self.next_group_id = 1  # Counter for assigning group IDs to duplicate sets
        
        # Mandala mode
        self.mandala_mode = False  # Whether to create 4 radial copies around center
        self.mandala_click_mode = False  # Whether to create mandala copies on each click
        
        # Paint mode
        self.paint_mode = False  # Whether to paint polygons with selected color on click

        # Eyedropper mode
        self.eyedropper_mode = False  # Whether to sample a polygon color on next click

        # Replace-color eyedropper mode
        self.replace_eyedropper_mode = False  # Whether to sample source color for replace-all
        
        # Random mode
        self.random_mode = False  # Whether to apply random variations in polygon drawing
        
        # Line mode
        self.line_mode = False  # Whether to draw lines and create square polygons along them
        self.line_start_point = None  # Starting point for line drawing
        self.current_line_end = None  # Current end point while dragging
        self.is_drawing_line = False  # Whether currently drawing a line
        self.line_points = []  # List of all points along the free-drawn line
        self.line_smoothing_factor = 0.05  # Default smoothing factor for splines
        self.line_polygon_size = 15  # Default size in pixels for line polygons
        self.num_parallel_lines = 0  # Number of parallel lines on each side
        self.line_polygon_gap = 2  # Gap between polygons in line mode
        self.parallel_line_gap = 2  # Gap between parallel lines
        
        # Overlap visualization
        self.showing_overlaps = False
        self.overlap_data = []  # List of (poly1_index, poly2_index, overlap_polygon) tuples
        
        # Center point offset
        self.center_offset_x = 0
        self.center_offset_y = 0
        
        # Selection tracking
        self.selected_polygon_index = -1  # Index of currently selected polygon (-1 means none)
        
        # Control point editing
        self.selected_control_point = -1  # Index of selected control point (-1 means none)
        self.is_dragging_control_point = False
        self.control_point_size = 8  # Size of control point circles in pixels
        
        # Polygon dragging
        self.is_dragging_polygon = False
        self.polygon_drag_start = None  # Starting position for polygon drag
        
        # Circle drawing
        # Image drag handle
        self.is_dragging_image = False
        self.image_drag_start_offset_x = 0
        self.image_drag_start_offset_y = 0
        self.drag_handle_size = 12  # Size of the drag handle circle
        
        # Image visibility
        self.show_image = True  # Default to showing image
        
        # Grid variables
        self.show_grid = False  # Whether to show the grid
        self.grid_size = 300  # Size of each individual grid box/cell in world coordinates (increased for 3x3)
        self.grid_offset_x = 0  # Grid offset in world coordinates
        self.grid_offset_y = 0  # Grid offset in world coordinates
        self.grid_dragging = False  # Whether we're dragging the grid
        self.grid_drag_start = None  # Starting point for grid drag
        self.grid_drag_world_start = None  # Starting world coordinates for grid drag
        
        # Drawing properties
        self.edge_width = 1.1  # Width for polygon edges
        self.background_color = QColor(255, 255, 255)  # Canvas background color
        
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
            
            # Exit line mode if active
            if self.line_mode:
                self.line_mode = False
                self.line_points = []
                self.is_drawing_line = False
                self.line_start_point = None
                self.current_line_end = None
                
                # Update line checkbox to reflect the change
                parent = self.parent()
                while parent:
                    for child in parent.findChildren(QCheckBox):
                        if child.text() == "Line" and hasattr(child, 'setChecked'):
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
    
    def set_image_visible(self, visible):
        """Set whether to show the background image"""
        self.show_image = visible
        self.update()  # Refresh display
    
    def set_line_mode(self, enabled):
        """Set whether line mode is enabled"""
        self.line_mode = enabled
        if enabled:
            # Clear any current line drawing state
            self.line_start_point = None
            self.current_line_end = None
            self.is_drawing_line = False
            self.line_points = []
            # Exit other modes if active
            if self.polygon_mode:
                self.polygon_mode = False
                self.polygon_points = []
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
        self.update()
    
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
            
            # Store the original image for scaling
            self.original_background_image = original_pixmap
            self.current_x_scale = 1.0  # Reset X scale to 100%
            self.current_y_scale = 1.0  # Reset Y scale to 100%
            
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
            
            # Center the image in the middle of the grid
            self.center_image_on_grid()
            
            self.update()  # Trigger repaint
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def center_image_on_grid(self):
        """Center the image in the middle of the grid"""
        if not self.background_image or self.background_image.isNull():
            return
        
        # Reset grid position to origin (0,0)
        self.grid_offset_x = 0
        self.grid_offset_y = 0
        
        # Get image dimensions
        image_width = self.background_image.width()
        image_height = self.background_image.height()
        
        # Calculate the center of the grid (3x3 grid starting at 0,0)
        # Grid center is at (1.5 * grid_size, 1.5 * grid_size)
        grid_center_x = self.grid_size * 1.5
        grid_center_y = self.grid_size * 1.5
        
        # Calculate image offset to center the image on the grid center
        # Image offset is the top-left corner of the image
        self.image_offset_x = grid_center_x - (image_width / 2)
        self.image_offset_y = grid_center_y - (image_height / 2)
    
    def scale_background_image(self, x_scale_factor, y_scale_factor=None):
        """Scale the background image by the given factors (1.0 = 100%)"""
        if not self.original_background_image or self.original_background_image.isNull():
            return
        
        # If only one scale factor provided, use it for both dimensions (backward compatibility)
        if y_scale_factor is None:
            y_scale_factor = x_scale_factor
        
        # Store current scales
        self.current_x_scale = x_scale_factor
        self.current_y_scale = y_scale_factor
        
        # Calculate new dimensions
        original_width = self.original_background_image.width()
        original_height = self.original_background_image.height()
        new_width = int(original_width * x_scale_factor)
        new_height = int(original_height * y_scale_factor)
        
        # Scale the image from the original
        self.background_image = self.original_background_image.scaled(
            new_width, new_height,
            Qt.IgnoreAspectRatio,  # Allow independent X/Y scaling
            Qt.SmoothTransformation
        )
        
        # Re-center the image on the grid after scaling
        self.center_image_on_grid()
        
        # Trigger repaint
        self.update()
    
    def toggle_polygon_mode(self):
        """Toggle polygon drawing mode on/off"""
        self.polygon_mode = not self.polygon_mode
        
        if self.polygon_mode:
            # Entering polygon mode - exit eraser mode if active
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
            
            # Exit line mode if active
            if self.line_mode:
                self.line_mode = False
                self.line_points = []
                self.is_drawing_line = False
                self.line_start_point = None
                self.current_line_end = None
                
                # Update line checkbox to reflect the change
                parent = self.parent()
                while parent:
                    for child in parent.findChildren(QCheckBox):
                        if child.text() == "Line" and hasattr(child, 'setChecked'):
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
            # Create only a single polygon
            self.create_single_polygon()
        # If not enough points, keep them - user might want to add more
    
    def create_single_polygon(self):
        """Create a single polygon and optionally its duplicates"""
        if len(self.polygon_points) < 3:
            return

        # Save state before adding new polygons
        self.save_state()

        # Generate unique group ID for this polygon and its copies (same method as line mode)
        current_group_id = max([p.get('group_id', 0) for p in self.polygons if p.get('group_id') is not None] + [0]) + 1
        
        print(f"DEBUG: Creating polygon mode polygon with group_id={current_group_id}")

        # Create original polygon with transparent fill and black frame
        original_polygon = {
            'points': list(self.polygon_points),  # Copy the points
            'color': QColor(0, 0, 0, 0),  # Transparent fill
            'frame_color': QColor(0, 0, 0, 255),  # Black frame
            'group_id': current_group_id  # Group ID for linking with copies
        }
        
        # Add original polygon
        self.polygons.append(original_polygon)
        
        # If duplicate mode is enabled, create 8 copies with offsets and colored frames
        if self.duplicate_mode:
            box_size = self.grid_size
            
            # Define the 8 offsets and corresponding frame colors
            offsets_and_colors = [
                ((-box_size, -box_size), QColor(255, 0, 0, 255)),    # 1st copy - Red frame
                ((-box_size, 0), QColor(0, 0, 255, 255)),            # 2nd copy - Blue frame
                ((-box_size, box_size), QColor(144, 238, 144, 255)), # 3rd copy - Light green frame
                ((0, -box_size), QColor(128, 0, 128, 255)),          # 4th copy - Purple frame
                ((0, box_size), QColor(255, 255, 0, 255)),           # 5th copy - Yellow frame
                ((box_size, -box_size), QColor(255, 192, 203, 255)), # 6th copy - Pink frame
                ((box_size, 0), QColor(128, 128, 128, 255)),         # 7th copy - Gray frame
                ((box_size, box_size), QColor(173, 216, 230, 255))   # 8th copy - Light blue frame
            ]
            
            # Create each duplicate with same group ID
            for (offset_x, offset_y), frame_color in offsets_and_colors:
                duplicate_points = []
                for point in self.polygon_points:
                    new_x = point[0] + offset_x
                    new_y = point[1] + offset_y
                    duplicate_points.append((new_x, new_y))
                
                duplicate_polygon = {
                    'points': duplicate_points,
                    'color': QColor(0, 0, 0, 0),  # Transparent fill
                    'frame_color': frame_color,   # Colored frame
                    'group_id': current_group_id  # Same group ID as original
                }
                
                self.polygons.append(duplicate_polygon)
        
        # If mandala mode is enabled, create 4 radial copies around center
        elif self.mandala_mode:
            self.create_mandala_copies(current_group_id)
        
        # Clear current points
        self.polygon_points = []
        self.update()  # Refresh display
    
    def create_mandala_copies(self, group_id):
        """Create 4 radial copies of the current polygon around grid center"""
        
        # Get grid center as mandala center (center of 3x3 grid)
        grid_center_x = self.grid_offset_x + (self.grid_size * 1.5)
        grid_center_y = self.grid_offset_y + (self.grid_size * 1.5)
        
        # Define 4 rotation angles (90 degrees each)
        angles_and_colors = [
            (90, QColor(255, 0, 0, 255)),    # 90° - Red frame
            (180, QColor(0, 255, 0, 255)),   # 180° - Green frame  
            (270, QColor(0, 0, 255, 255))    # 270° - Blue frame
        ]
        
        # Create each rotated copy
        for angle_degrees, frame_color in angles_and_colors:
            angle_radians = math.radians(angle_degrees)
            
            # Rotate each point around the grid center
            rotated_points = []
            for world_x, world_y in self.polygon_points:
                # Translate to origin (relative to center)
                rel_x = world_x - grid_center_x
                rel_y = world_y - grid_center_y
                
                # Apply rotation
                rotated_x = rel_x * math.cos(angle_radians) - rel_y * math.sin(angle_radians)
                rotated_y = rel_x * math.sin(angle_radians) + rel_y * math.cos(angle_radians)
                
                # Translate back
                final_x = rotated_x + grid_center_x
                final_y = rotated_y + grid_center_y
                
                rotated_points.append((final_x, final_y))
            
            # Create the rotated polygon copy
            mandala_polygon = {
                'points': rotated_points,
                'color': QColor(0, 0, 0, 0),  # Transparent fill
                'frame_color': frame_color,   # Colored frame
                'group_id': group_id,         # Same group ID as original
                'rotation_angle': angle_degrees
            }
            
            self.polygons.append(mandala_polygon)

    def create_mandala_click_copies(self, world_x, world_y):
        """Create mandala copies of an existing polygon at the clicked point"""
        # Find the polygon at the clicked point
        clicked_polygon_index = -1
        for i, polygon_data in enumerate(self.polygons):
            if self.point_in_polygon(world_x, world_y, polygon_data['points']):
                clicked_polygon_index = i
                break
        
        if clicked_polygon_index == -1:
            return  # No polygon found at click point
        
        # Get the clicked polygon
        source_polygon = self.polygons[clicked_polygon_index]
        
        # Save state before adding new polygons
        self.save_state()
        
        # Generate unique group ID for this mandala set
        current_group_id = max([p.get('group_id', 0) for p in self.polygons if p.get('group_id') is not None] + [0]) + 1
        
        print(f"DEBUG: Creating mandala click copies with group_id={current_group_id}")
        
        # Update the source polygon to be part of the mandala group
        self.polygons[clicked_polygon_index]['group_id'] = current_group_id
        self.polygons[clicked_polygon_index]['rotation_angle'] = 0  # Mark as original
        
        # Mark the clicked polygon with green fill to indicate it was used in mandala click mode
        self.polygons[clicked_polygon_index]['color'] = QColor(0, 255, 0, 128)  # Green with transparency
        self.polygons[clicked_polygon_index]['mandala_clicked'] = True  # Mark for later identification
        
        # Get grid center as mandala center
        grid_center_x = self.grid_offset_x + (self.grid_size * 1.5)
        grid_center_y = self.grid_offset_y + (self.grid_size * 1.5)
        
        # Define 3 rotation angles and colors for the copies
        angles_and_colors = [
            (90, QColor(255, 0, 0, 255)),    # 90° - Red frame
            (180, QColor(0, 255, 0, 255)),   # 180° - Green frame  
            (270, QColor(0, 0, 255, 255))    # 270° - Blue frame
        ]
        
        # Create each rotated copy
        for angle_degrees, frame_color in angles_and_colors:
            angle_radians = math.radians(angle_degrees)
            
            # Rotate each point around the grid center
            rotated_points = []
            for world_x_pt, world_y_pt in source_polygon['points']:
                # Translate to origin (relative to grid center)
                rel_x = world_x_pt - grid_center_x
                rel_y = world_y_pt - grid_center_y
                
                # Apply rotation
                rotated_x = rel_x * math.cos(angle_radians) - rel_y * math.sin(angle_radians)
                rotated_y = rel_x * math.sin(angle_radians) + rel_y * math.cos(angle_radians)
                
                # Translate back
                final_x = rotated_x + grid_center_x
                final_y = rotated_y + grid_center_y
                
                rotated_points.append((final_x, final_y))
            
            # Create the rotated polygon copy
            mandala_copy = {
                'points': rotated_points,
                'color': QColor(0, 0, 0, 0),  # Keep transparent fill for copies
                'frame_color': frame_color,   # Colored frame for identification
                'group_id': current_group_id, # Same group ID as original
                'rotation_angle': angle_degrees
            }
            
            self.polygons.append(mandala_copy)
        
        self.update()  # Refresh display

    def sample_polygon_color_at_point(self, world_x, world_y):
        """Sample the color of the polygon at the clicked point and send it to the left panel."""
        self.eyedropper_mode = False
        self.setCursor(Qt.ArrowCursor)
        for polygon_data in self.polygons:
            if self.point_in_polygon(world_x, world_y, polygon_data['points']):
                color = polygon_data.get('color')
                if color is not None and hasattr(self, 'left_panel'):
                    self.left_panel.receive_eyedropper_color(QColor(color))
                return

    def sample_replace_source_at_point(self, world_x, world_y):
        """Sample the source color for replace-all and send it to the left panel."""
        self.replace_eyedropper_mode = False
        self.setCursor(Qt.ArrowCursor)
        candidates = []
        for i, polygon_data in enumerate(self.polygons):
            if self.point_in_polygon(world_x, world_y, polygon_data['points']):
                area = self.calculate_polygon_area(polygon_data['points'])
                candidates.append((i, area))
        if candidates:
            candidates.sort(key=lambda x: (x[1], -x[0]))
            color = self.polygons[candidates[0][0]].get('color')
            if color is not None and hasattr(self, 'left_panel'):
                self.left_panel.receive_replace_source_color(QColor(color))

    def paint_polygon_at_point(self, world_x, world_y):
        """Paint a polygon at the clicked point with the selected color"""
        # Find the topmost (smallest area) polygon at the clicked point
        candidates = []
        for i, polygon_data in enumerate(self.polygons):
            if self.point_in_polygon(world_x, world_y, polygon_data['points']):
                area = self.calculate_polygon_area(polygon_data['points'])
                candidates.append((i, area))

        if not candidates:
            return  # No polygon found at click point

        # Pick smallest polygon (same logic as select_polygon_at_point)
        candidates.sort(key=lambda x: (x[1], -x[0]))
        clicked_polygon_index = candidates[0][0]
        
        # Save state before painting
        self.save_state()
        
        # Get the selected color from the side panel
        selected_color = None
        # Check left panel first since that's where the color palette is
        if hasattr(self, 'left_panel') and hasattr(self.left_panel, 'selected_color'):
            selected_color = self.left_panel.selected_color
        elif hasattr(self, 'right_panel') and hasattr(self.right_panel, 'selected_color'):
            selected_color = self.right_panel.selected_color
        
        if selected_color is None:
            selected_color = QColor(0, 0, 0)  # Default to black
        
        # Paint the clicked polygon
        self.polygons[clicked_polygon_index]['color'] = QColor(selected_color)
        
        # If duplicate mode is enabled, also paint polygons in the same group
        if self.duplicate_mode:
            clicked_group_id = self.polygons[clicked_polygon_index].get('group_id')
            if clicked_group_id is not None:
                for polygon in self.polygons:
                    if polygon.get('group_id') == clicked_group_id:
                        polygon['color'] = QColor(selected_color)
        
        self.update()  # Refresh display

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
    
    def show_overlaps(self):
        """Toggle overlap visualization"""
        self.showing_overlaps = not self.showing_overlaps
        if self.showing_overlaps:
            self.detect_overlaps()
        else:
            self.overlap_data = []
        self.update()
    
    def detect_overlaps(self):
        """Detect all polygon overlaps and store them for visualization, including frame overlaps"""
        from shapely.geometry import Polygon
        from shapely.strtree import STRtree
        import time
        
        start_time = time.time()
        self.overlap_data = []
        
        if len(self.polygons) < 2:
            return
        
        # Buffer amount is half the edge width to account for frame thickness
        buffer_amount = self.edge_width / 2.0
        
        # Create valid shapely polygons with original indices
        valid_polygons = []
        original_indices = []
        
        for i, polygon_data in enumerate(self.polygons):
            try:
                poly_points = polygon_data['points']
                if len(poly_points) < 3:
                    continue
                    
                poly = Polygon(poly_points)
                if poly.is_valid:
                    # Buffer polygon to account for frame thickness
                    poly_buffered = poly.buffer(buffer_amount)
                    valid_polygons.append(poly_buffered)
                    original_indices.append(i)
            except Exception:
                continue
        
        if len(valid_polygons) < 2:
            return
        
        # Create spatial index (R-tree) for efficient overlap detection
        tree = STRtree(valid_polygons)
        
        processed_pairs = set()
        
        # Use spatial index for efficient overlap detection
        for i_valid, poly1_buffered in enumerate(valid_polygons):
            # Query spatial index for potential overlaps
            possible_matches = tree.query(poly1_buffered)
            
            for j_valid in possible_matches:
                if i_valid == j_valid:
                    continue
                
                # Get original polygon indices
                i_original = original_indices[i_valid]
                j_original = original_indices[j_valid]
                
                # Avoid duplicate checks
                pair = tuple(sorted((i_original, j_original)))
                if pair in processed_pairs:
                    continue
                
                processed_pairs.add(pair)
                
                try:
                    poly2_buffered = valid_polygons[j_valid]
                    
                    # Check for actual intersection (simplified - just like mosaic editor)
                    if poly1_buffered.intersects(poly2_buffered):
                        intersection = poly1_buffered.intersection(poly2_buffered)
                        
                        # Only consider significant overlaps (simplified check)
                        if hasattr(intersection, 'area') and intersection.area > 0.01:
                            # Store simple overlap data (no complex geometry)
                            self.overlap_data.append((i_original, j_original, []))
                                        
                except Exception:
                    # Skip problematic polygon pairs
                    continue
        
        end_time = time.time()
        print(f"Overlap detection completed in {end_time - start_time:.2f} seconds")
        print(f"Found {len(self.overlap_data)} overlapping pairs")
    
    def polygons_overlap_simple(self, points1, points2):
        """Simple overlap detection using point-in-polygon tests (fallback method)"""
        # Check if any vertex of poly1 is inside poly2
        for point in points1:
            if self.point_in_polygon(point[0], point[1], points2):
                return True
        
        # Check if any vertex of poly2 is inside poly1
        for point in points2:
            if self.point_in_polygon(point[0], point[1], points1):
                return True
                
        # TODO: Could add edge-edge intersection checks for more thorough detection
        return False
    
    def delete_green_polygons(self):
        """Delete all green (older) overlapping polygons"""
        if not self.showing_overlaps or not self.overlap_data:
            # First ensure overlaps are detected
            self.detect_overlaps()
            if not self.overlap_data:
                return  # No overlaps found
        
        # Save state before making changes
        self.save_state()
        
        # Collect indices of polygons to delete (older ones in overlaps)
        to_delete = set()
        for i, j, overlap_points in self.overlap_data:
            # i is the older polygon (lower index), j is newer
            to_delete.add(i)
        
        # Delete polygons in reverse order to maintain correct indices
        for index in sorted(to_delete, reverse=True):
            if 0 <= index < len(self.polygons):
                del self.polygons[index]
        
        # Clear overlap data since polygons have changed
        self.overlap_data = []
        self.showing_overlaps = False
        
        # Update display
        self.update()
        print(f"Deleted {len(to_delete)} green (older) overlapping polygons")
    
    def delete_red_polygons(self):
        """Delete all red (newer) overlapping polygons"""
        if not self.showing_overlaps or not self.overlap_data:
            # First ensure overlaps are detected
            self.detect_overlaps()
            if not self.overlap_data:
                return  # No overlaps found
        
        # Save state before making changes
        self.save_state()
        
        # Collect indices of polygons to delete (newer ones in overlaps)
        to_delete = set()
        for i, j, overlap_points in self.overlap_data:
            # i is the older polygon (lower index), j is newer
            to_delete.add(j)
        
        # Delete polygons in reverse order to maintain correct indices
        for index in sorted(to_delete, reverse=True):
            if 0 <= index < len(self.polygons):
                del self.polygons[index]
        
        # Clear overlap data since polygons have changed
        self.overlap_data = []
        self.showing_overlaps = False
        
        # Update display
        self.update()
        print(f"Deleted {len(to_delete)} red (newer) overlapping polygons")
    
    def save_state(self):
        """Save the current polygon state to the undo stack"""
        import copy
        
        # Create a deep copy of the current polygons list
        current_state = copy.deepcopy(self.polygons)
        
        # Add to undo stack
        self.undo_stack.append(current_state)
        
        # Limit the size of the undo stack
        if len(self.undo_stack) > self.max_undo_states:
            self.undo_stack.pop(0)  # Remove oldest state
    
    def undo_last_action(self):
        """Undo the last action by restoring the previous polygon state"""
        if not self.undo_stack:
            print("No actions to undo")
            return False
        
        # Restore the last saved state
        previous_state = self.undo_stack.pop()
        self.polygons = previous_state
        
        # Clear overlap data since polygons have changed
        self.overlap_data = []
        self.showing_overlaps = False
        
        # Update display
        self.update()
        print(f"Undid last action. {len(self.undo_stack)} undo states remaining.")
        return True
    
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        # Ensure canvas has focus for keyboard events
        self.setFocus()
        
        if event.button() == Qt.LeftButton and self.line_mode:
            # In line mode, left click starts line drawing
            world_x, world_y = self.screen_to_world(event.x(), event.y())
            self.line_start_point = (world_x, world_y)
            self.current_line_end = (world_x, world_y)
            self.is_drawing_line = True
            self.line_points = [(world_x, world_y)]  # Start with the first point
        elif event.button() == Qt.LeftButton and self.polygon_mode:
            # In polygon mode, left click adds point to polygon
            self.add_polygon_point(event.x(), event.y())
        elif event.button() == Qt.RightButton and self.polygon_mode:
            # Right click finishes the polygon
            self.finish_polygon()
        elif event.button() == Qt.RightButton and not self.polygon_mode:
            pass  # reserved for future use
        elif event.button() == Qt.LeftButton and not self.polygon_mode:
            # Eyedropper: sample polygon color if active (before all other handlers)
            if self.replace_eyedropper_mode:
                world_x, world_y = self.screen_to_world(event.x(), event.y())
                self.sample_replace_source_at_point(world_x, world_y)
                return
            if self.eyedropper_mode:
                world_x, world_y = self.screen_to_world(event.x(), event.y())
                self.sample_polygon_color_at_point(world_x, world_y)
                return

            # Check for eraser mode first
            if self.eraser_mode:
                world_x, world_y = self.screen_to_world(event.x(), event.y())
                self.erase_polygon_at_point(world_x, world_y)
                self.is_erasing = True  # Start erasing mode for drag
                return
            
            # Check for image drag handle click
            if self.is_point_in_image_drag_handle(event.x(), event.y()):
                # Start dragging image
                self.is_dragging_image = True
                self.image_drag_start_offset_x = self.image_offset_x
                self.image_drag_start_offset_y = self.image_offset_y
                self.last_pan_point = event.pos()
                self.setCursor(Qt.ClosedHandCursor)
                return
            
            # Check for grid drag handle click
            elif self.is_point_in_grid_drag_handle(event.x(), event.y()):
                # Start dragging grid
                self.grid_dragging = True
                self.grid_drag_start = event.pos()
                world_x, world_y = self.screen_to_world(event.x(), event.y())
                self.grid_drag_world_start = (world_x, world_y)
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
                # Check for polygon selection, mandala click, paint, or start dragging
                world_x, world_y = self.screen_to_world(event.x(), event.y())
                
                # If mandala click mode is enabled, create mandala copies of clicked polygon
                if self.mandala_click_mode:
                    self.create_mandala_click_copies(world_x, world_y)
                # If paint mode is enabled, paint the clicked polygon
                elif self.paint_mode:
                    self.paint_polygon_at_point(world_x, world_y)
                else:
                    # Check if clicking on an already selected polygon to start dragging
                    if (self.selected_polygon_index >= 0 and 
                        self.selected_polygon_index < len(self.polygons) and
                        self.point_in_polygon(world_x, world_y, self.polygons[self.selected_polygon_index]['points'])):
                        # Start dragging the selected polygon
                        self.is_dragging_polygon = True
                        self.polygon_drag_start = (world_x, world_y)
                        self.setCursor(Qt.ClosedHandCursor)
                    else:
                        # Select polygon at point (or deselect if clicking empty space)
                        self.select_polygon_at_point(world_x, world_y)
        elif event.button() == Qt.MiddleButton:
            # Start panning
            self.is_panning = True
            self.last_pan_point = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events"""
        if self.is_drawing_line:
            # Add points along the path for free drawing
            world_x, world_y = self.screen_to_world(event.x(), event.y())
            self.current_line_end = (world_x, world_y)
            
            # Add point to path if it's far enough from the last point to avoid too many points
            if len(self.line_points) == 0:
                self.line_points.append((world_x, world_y))
            else:
                last_x, last_y = self.line_points[-1]
                distance = math.sqrt((world_x - last_x)**2 + (world_y - last_y)**2)
                # Only add point if it's at least 5 world units away from last point
                if distance > 5:
                    self.line_points.append((world_x, world_y))
            
            self.update()  # Refresh to show the line
        elif self.is_erasing:
            # Continue erasing while dragging
            world_x, world_y = self.screen_to_world(event.x(), event.y())
            self.erase_polygon_at_point(world_x, world_y)
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
        elif self.grid_dragging and self.grid_drag_start:
            # Drag grid by updating offset
            delta = event.pos() - self.grid_drag_start
            # Convert screen delta to world delta (accounting for zoom)
            world_delta_x = delta.x() / self.zoom_factor
            world_delta_y = delta.y() / self.zoom_factor
            
            # Calculate new grid position
            start_world_x, start_world_y = self.grid_drag_world_start
            new_world_x = start_world_x + world_delta_x
            new_world_y = start_world_y + world_delta_y
            
            # Update grid offset to align with new position
            self.grid_offset_x = new_world_x
            self.grid_offset_y = new_world_y
            self.update()
            
        elif self.is_dragging_polygon and self.polygon_drag_start:
            # Drag polygon by calculating movement delta
            world_x, world_y = self.screen_to_world(event.x(), event.y())
            start_x, start_y = self.polygon_drag_start
            
            # Calculate movement delta
            delta_x = world_x - start_x
            delta_y = world_y - start_y
            
            # Move the selected polygon and its group
            if self.selected_polygon_index >= 0:
                self.move_polygon_group(self.selected_polygon_index, delta_x, delta_y)
                
                # Update drag start point for next movement
                self.polygon_drag_start = (world_x, world_y)
                self.update()
            
        elif self.is_dragging_control_point and self.selected_control_point >= 0:
            # Drag control point to reshape polygon and optionally its group
            world_x, world_y = self.screen_to_world(event.x(), event.y())
            
            # Update the control point position of the selected polygon and its group
            if (self.selected_polygon_index >= 0 and 
                self.selected_polygon_index < len(self.polygons)):
                
                selected_polygon = self.polygons[self.selected_polygon_index]
                group_id = selected_polygon.get('group_id')
                selected_points = selected_polygon['points']
                
                if self.selected_control_point < len(selected_points):
                    # During dragging, only update the selected polygon
                    selected_points[self.selected_control_point] = (world_x, world_y)
                    self.update()
                    
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
                
                if (self.background_image and 
                      self.is_point_in_image_drag_handle(event.x(), event.y())):
                    self.setCursor(Qt.OpenHandCursor)
                elif self.is_point_in_grid_drag_handle(event.x(), event.y()):
                    self.setCursor(Qt.OpenHandCursor)
                elif not self.polygon_mode and not self.eraser_mode:
                    self.setCursor(Qt.ArrowCursor)
                elif self.eraser_mode and not self.polygon_mode:
                    self.setCursor(Qt.PointingHandCursor)
        
        # Update cursor position in right panel (for all mouse moves)
        if hasattr(self, 'right_panel') and self.right_panel:
            world_x, world_y = self.screen_to_world(event.x(), event.y())
            self.right_panel.update_cursor_position(world_x, world_y)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if self.line_mode and self.is_drawing_line:
            # Finish line drawing and create square polygons along the line
            self.create_trapezoid_polygons_along_line()
            self.is_drawing_line = False
            self.current_line_end = None
            self.line_start_point = None
            self.line_points = []  # Clear the line points
            self.update()
        elif self.is_erasing:
            # Stop erasing
            self.is_erasing = False
        elif self.is_dragging_control_point:
            # Update all copies when control point dragging is complete (only if duplicate mode is active)
            if (self.duplicate_mode and self.selected_polygon_index >= 0 and 
                self.selected_polygon_index < len(self.polygons)):
                
                selected_polygon = self.polygons[self.selected_polygon_index]
                group_id = selected_polygon.get('group_id')
                
                # Synchronize group polygons only when duplicate mode is enabled
                if group_id is not None:
                    self.update_group_control_points_after_drag(group_id)
            
            # Stop dragging control point
            self.is_dragging_control_point = False
            self.selected_control_point = -1
            self.setCursor(Qt.ArrowCursor)
        elif self.is_dragging_image:
            self.is_dragging_image = False
            self.last_pan_point = None
            self.setCursor(Qt.ArrowCursor)
        elif self.grid_dragging:
            # Stop dragging grid
            self.grid_dragging = False
            self.grid_drag_start = None
            self.grid_drag_world_start = None
            self.setCursor(Qt.ArrowCursor)
        elif self.is_dragging_polygon:
            # Stop dragging polygon
            self.is_dragging_polygon = False
            self.polygon_drag_start = None
            self.setCursor(Qt.ArrowCursor)
        elif self.is_panning:
            self.is_panning = False
            self.last_pan_point = None
            self.setCursor(Qt.ArrowCursor if not self.polygon_mode else Qt.BlankCursor)
    
    def create_smooth_spline(self, points, num_points=None):
        """Create a smooth spline from the given points"""
        if len(points) < 3:
            return points, None  # Need at least 3 points for spline
            
        try:
            # Convert points to numpy arrays for x and y coordinates
            x_coords = np.array([p[0] for p in points])
            y_coords = np.array([p[1] for p in points])
            
            # Create parametric spline representation
            # s controls smoothing: 0=interpolate exactly, higher values=more smoothing
            # k=3 means cubic spline (can be 1-5, 3 is good for smoothness)
            
            # Calculate smoothing factor based on number of points and their spread
            # More aggressive smoothing for hand-drawn lines
            num_points_factor = len(points)
            coordinate_range = max(max(x_coords) - min(x_coords), max(y_coords) - min(y_coords))
            smoothing_factor = coordinate_range * num_points_factor * self.line_smoothing_factor
            
            tck, u = splprep([x_coords, y_coords], s=smoothing_factor, k=min(3, len(points)-1))
            
            # If num_points not specified, use 5x the original number of points for smoothness
            if num_points is None:
                num_points = len(points) * 5
                
            # Generate evenly spaced parameters for smooth curve
            u_new = np.linspace(0, 1, num_points)
            
            # Evaluate the spline at the new parameters
            smooth_coords = splev(u_new, tck)
            
            # Convert back to list of tuples
            smooth_points = list(zip(smooth_coords[0], smooth_coords[1]))
            
            return smooth_points, (tck, u_new)
            
        except Exception as e:
            print(f"Spline creation failed: {e}")
            return points, None  # Return original points if spline fails
    
    def get_spline_angle_at_parameter(self, tck, u_param):
        """Calculate the tangent angle of the spline at a given parameter u"""
        try:
            # Calculate first derivative (tangent vector) at the parameter
            derivatives = splev(u_param, tck, der=1)
            dx_du = derivatives[0]
            dy_du = derivatives[1]
            
            # Calculate angle from tangent vector
            angle = math.atan2(dy_du, dx_du)
            return angle
            
        except Exception as e:
            print(f"Angle calculation failed: {e}")
            return 0  # Return 0 angle if calculation fails
    
    def create_trapezoid_polygons_along_line(self):
        """Create trapezoid polygons along the drawn line and parallel lines on both sides"""
        if len(self.line_points) < 2:
            return
        
        # Save state before creating line polygons
        self.save_state()
            
        # Create smooth spline from the original line points
        smooth_points, spline_data = self.create_smooth_spline(self.line_points)
        
        if len(smooth_points) < 2:
            return
            
        # Create polygons along the main line and all parallel lines
        self.create_polygons_along_single_line(smooth_points, spline_data, 0)  # Main line (offset = 0)
        
        # Create parallel lines on both sides
        for line_number in range(1, self.num_parallel_lines + 1):
            offset_distance = (self.line_polygon_size + self.parallel_line_gap) * line_number  # Configurable gap between parallel lines
            
            # Create line on positive side (right side when walking along the path)
            self.create_polygons_along_single_line(smooth_points, spline_data, offset_distance)
            
            # Create line on negative side (left side when walking along the path)
            self.create_polygons_along_single_line(smooth_points, spline_data, -offset_distance)
    
    def create_polygons_along_single_line(self, smooth_points, spline_data, offset_distance):
        """Create trapezoid polygons along a single line (original or parallel offset line)"""
        # Use configurable square size directly - painter scaling handles zoom
        square_size = self.line_polygon_size
        half_size = square_size / 2
        
        # If offset_distance is not 0, we need to create offset smooth points
        if offset_distance != 0:
            offset_smooth_points = self.create_offset_line(smooth_points, spline_data, offset_distance)
            if len(offset_smooth_points) < 2:
                return
        else:
            offset_smooth_points = smooth_points
        
        # Calculate total path length and segment lengths for the (possibly offset) line
        total_path_length = 0
        segment_lengths = []
        for i in range(len(offset_smooth_points) - 1):
            x1, y1 = offset_smooth_points[i]
            x2, y2 = offset_smooth_points[i + 1]
            segment_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            segment_lengths.append(segment_length)
            total_path_length += segment_length
        
        if total_path_length == 0:
            return
            
        # Use distance-based spacing for consistent polygon sizes
        gap_screen = self.line_polygon_gap  # Configurable gap between polygons
        step_size = square_size + gap_screen  # Distance between polygon centers
        
        # But each trapezoid should be smaller than step_size to create the gap
        trapezoid_width = square_size  # Keep trapezoid width equal to square_size
        
        num_polygons = int(total_path_length / step_size)
        
        if num_polygons == 0:
            return
            
        # Helper function to find position and angle at a given distance along the offset line
        def get_position_and_angle_at_distance(target_distance):
            current_distance = 0
            for segment_index in range(len(segment_lengths)):
                segment_end_distance = current_distance + segment_lengths[segment_index]
                
                if target_distance <= segment_end_distance:
                    segment_progress = (target_distance - current_distance) / segment_lengths[segment_index]
                    
                    # Interpolate position along the offset line
                    x1, y1 = offset_smooth_points[segment_index]
                    x2, y2 = offset_smooth_points[segment_index + 1]
                    center_x = x1 + segment_progress * (x2 - x1)
                    center_y = y1 + segment_progress * (y2 - y1)
                    
                    # Calculate angle using the original spline data
                    if spline_data is not None:
                        tck, u_new = spline_data
                        u_param = (segment_index + segment_progress) / len(smooth_points)
                        angle = self.get_spline_angle_at_parameter(tck, u_param)
                    else:
                        # Use direction of the original line for angle
                        orig_x1, orig_y1 = smooth_points[segment_index]
                        orig_x2, orig_y2 = smooth_points[segment_index + 1]
                        dx = orig_x2 - orig_x1
                        dy = orig_y2 - orig_y1
                        angle = math.atan2(dy, dx)
                    
                    return center_x, center_y, angle
                    
                current_distance = segment_end_distance
            
            # If we reach here, return the last point
            last_point = offset_smooth_points[-1]
            if spline_data is not None:
                tck, u_new = spline_data
                last_angle = self.get_spline_angle_at_parameter(tck, 1.0)
            else:
                dx = smooth_points[-1][0] - smooth_points[-2][0]
                dy = smooth_points[-1][1] - smooth_points[-2][1]
                last_angle = math.atan2(dy, dx)
            return last_point[0], last_point[1], last_angle
        
        # Create trapezoids using distance-based spacing
        for polygon_index in range(num_polygons):
            # Calculate center distance for this polygon
            center_distance = (polygon_index + 0.5) * step_size
            
            if center_distance >= total_path_length:
                break
                
            # Get position and angle at center
            center_x, center_y, center_angle = get_position_and_angle_at_distance(center_distance)
            
            # Calculate start and end distances for the trapezoid
            half_trapezoid_width = trapezoid_width / 2
            start_distance = max(0, center_distance - half_trapezoid_width)
            end_distance = min(total_path_length, center_distance + half_trapezoid_width)
            
            # Get positions and angles at start and end
            start_x, start_y, start_angle = get_position_and_angle_at_distance(start_distance)
            end_x, end_y, end_angle = get_position_and_angle_at_distance(end_distance)
            
            # Apply random size variation only if random mode is enabled
            if self.random_mode:
                # Random size variation: 80% to 100% of original size
                size_variation = random.uniform(0.8, 1.0)
                random_half_size = half_size * size_variation
            else:
                # Use original size without variation
                random_half_size = half_size
            
            # Create perpendicular lines at start and end positions
            start_perp_angle = start_angle + math.pi/2
            start_cos_perp = math.cos(start_perp_angle)
            start_sin_perp = math.sin(start_perp_angle)
            line_start_p1 = (start_x + start_cos_perp * random_half_size, start_y + start_sin_perp * random_half_size)
            line_start_p2 = (start_x - start_cos_perp * random_half_size, start_y - start_sin_perp * random_half_size)
            
            end_perp_angle = end_angle + math.pi/2
            end_cos_perp = math.cos(end_perp_angle)
            end_sin_perp = math.sin(end_perp_angle)
            line_end_p1 = (end_x + end_cos_perp * random_half_size, end_y + end_sin_perp * random_half_size)
            line_end_p2 = (end_x - end_cos_perp * random_half_size, end_y - end_sin_perp * random_half_size)
            
            # Create trapezoid from the 4 endpoints of the two lines
            polygon_points = [
                line_start_p1,
                line_start_p2, 
                line_end_p2,
                line_end_p1
            ]
            
            # Apply random movement to control points only if random mode is enabled
            if self.random_mode:
                # Add random movement to each control point (-1 to +1 pixels on X and Y)
                randomized_polygon_points = []
                for point in polygon_points:
                    x, y = point
                    random_x_offset = random.uniform(-1, 1)
                    random_y_offset = random.uniform(-1, 1)
                    randomized_point = (x + random_x_offset, y + random_y_offset)
                    randomized_polygon_points.append(randomized_point)
                final_polygon_points = randomized_polygon_points
            else:
                # Use original points without randomization
                final_polygon_points = polygon_points
            
            # Create polygon data
            polygon_data = {
                'points': final_polygon_points,
                'color': QColor(0, 0, 0, 0),  # Transparent fill
                'frame_color': QColor(0, 0, 0, 255),  # Black frame
                'group_id': None  # Line polygons are not grouped by default
            }
            
            # Add original polygon
            self.polygons.append(polygon_data)
            
            # If duplicate mode is enabled, create 8 copies with offsets and colored frames
            if self.duplicate_mode:
                box_size = self.grid_size
                
                # Generate unique group ID for this polygon and its copies
                current_group_id = max([p.get('group_id', 0) for p in self.polygons if p.get('group_id') is not None] + [0]) + 1
                polygon_data['group_id'] = current_group_id  # Update original with group ID
                
                print(f"DEBUG: Creating line mode polygon with group_id={current_group_id}")
                
                # Define the 8 offsets and corresponding frame colors
                offsets_and_colors = [
                    ((-box_size, -box_size), QColor(255, 0, 0, 255)),    # 1st copy - Red frame
                    ((-box_size, 0), QColor(0, 0, 255, 255)),            # 2nd copy - Blue frame
                    ((-box_size, box_size), QColor(144, 238, 144, 255)), # 3rd copy - Light green frame
                    ((0, -box_size), QColor(128, 0, 128, 255)),          # 4th copy - Purple frame
                    ((0, box_size), QColor(255, 255, 0, 255)),           # 5th copy - Yellow frame
                    ((box_size, -box_size), QColor(255, 192, 203, 255)), # 6th copy - Pink frame
                    ((box_size, 0), QColor(128, 128, 128, 255)),         # 7th copy - Gray frame
                    ((box_size, box_size), QColor(173, 216, 230, 255))   # 8th copy - Light blue frame
                ]
                
                # Create each duplicate with same group ID
                for (offset_x, offset_y), frame_color in offsets_and_colors:
                    duplicate_points = []
                    for point in final_polygon_points:
                        new_x = point[0] + offset_x
                        new_y = point[1] + offset_y
                        duplicate_points.append((new_x, new_y))
                    
                    duplicate_polygon = {
                        'points': duplicate_points,
                        'color': QColor(0, 0, 0, 0),  # Transparent fill
                        'frame_color': frame_color,   # Colored frame
                        'group_id': current_group_id  # Same group ID as original
                    }
                    
                    self.polygons.append(duplicate_polygon)
    
    def create_offset_line(self, smooth_points, spline_data, offset_distance):
        """Create an offset line parallel to the original line at the given distance"""
        offset_points = []
        
        for i in range(len(smooth_points)):
            x, y = smooth_points[i]
            
            # Get the angle of the original line at this point
            if spline_data is not None:
                tck, u_new = spline_data
                u_param = i / (len(smooth_points) - 1) if len(smooth_points) > 1 else 0
                angle = self.get_spline_angle_at_parameter(tck, u_param)
            else:
                # Calculate angle from adjacent points
                if i == 0 and len(smooth_points) > 1:
                    dx = smooth_points[1][0] - smooth_points[0][0]
                    dy = smooth_points[1][1] - smooth_points[0][1]
                elif i == len(smooth_points) - 1 and len(smooth_points) > 1:
                    dx = smooth_points[i][0] - smooth_points[i-1][0]
                    dy = smooth_points[i][1] - smooth_points[i-1][1]
                elif len(smooth_points) > 2:
                    # Average of forward and backward directions
                    dx1 = smooth_points[i][0] - smooth_points[i-1][0]
                    dy1 = smooth_points[i][1] - smooth_points[i-1][1]
                    dx2 = smooth_points[i+1][0] - smooth_points[i][0]
                    dy2 = smooth_points[i+1][1] - smooth_points[i][1]
                    dx = (dx1 + dx2) / 2
                    dy = (dy1 + dy2) / 2
                else:
                    dx, dy = 1, 0  # Default direction
                angle = math.atan2(dy, dx)
            
            # Calculate perpendicular angle (90 degrees to the right)
            perp_angle = angle + math.pi/2
            
            # Calculate offset position
            offset_x = x + offset_distance * math.cos(perp_angle)
            offset_y = y + offset_distance * math.sin(perp_angle)
            
            offset_points.append((offset_x, offset_y))
        
        return offset_points
    
    def update_group_control_points_after_drag(self, group_id):
        """Update control points of all copies in the group based on the original polygon's new position"""
        if group_id is None or self.selected_polygon_index < 0:
            return
            
        print(f"DEBUG: Control point sync called - group_id={group_id}, selected_index={self.selected_polygon_index}, total_polygons={len(self.polygons)}")
            
        # Find all polygons in the group
        group_polygons = [p for p in self.polygons if p.get('group_id') == group_id]
        if len(group_polygons) < 2:  # Need at least 2 polygons to synchronize
            return
            
        # Get the dragged polygon (the one we just moved)
        if self.selected_polygon_index >= len(self.polygons):
            print(f"DEBUG: ERROR - selected_polygon_index {self.selected_polygon_index} >= total polygons {len(self.polygons)}")
            return
        dragged_polygon = self.polygons[self.selected_polygon_index]
        
        print(f"DEBUG: Dragged polygon group_id={dragged_polygon.get('group_id')}, expected group_id={group_id}")
        
        # CRITICAL CHECK: Verify the dragged polygon is actually in the group we expect
        if dragged_polygon.get('group_id') != group_id:
            print(f"DEBUG: ERROR - Dragged polygon is in wrong group! Expected {group_id}, got {dragged_polygon.get('group_id')}")
            return
        
        if self.selected_control_point < 0 or self.selected_control_point >= len(dragged_polygon['points']):
            return
            
        # Find the original polygon (the one with black frame)
        original_polygon = None
        for polygon in group_polygons:
            frame_color = polygon.get('frame_color', QColor(0, 0, 0, 255))
            if (frame_color.red() == 0 and frame_color.green() == 0 and 
                frame_color.blue() == 0 and frame_color.alpha() == 255):
                original_polygon = polygon
                break
                
        if not original_polygon:
            return
        
        # Define the box size (needed for offset calculations)
        box_size = self.grid_size
            
        # Get the new position of the moved control point from whichever polygon was dragged
        dragged_new_point = dragged_polygon['points'][self.selected_control_point]
        
        # Calculate where the original should be based on the dragged polygon's position
        if dragged_polygon is original_polygon:
            # If we dragged the original, use its position directly
            original_new_point = dragged_new_point
        else:
            # If we dragged a copy, calculate where the original should be
            # Find which copy this is by its frame color
            dragged_frame_color = dragged_polygon.get('frame_color', QColor(0, 0, 0, 255))
            
            # Map frame colors to their corresponding offsets
            color_to_offset = {
                (255, 0, 0): (-box_size, -box_size),     # Red
                (0, 0, 255): (-box_size, 0),             # Blue  
                (144, 238, 144): (-box_size, box_size),  # Light green
                (128, 0, 128): (0, -box_size),           # Purple
                (255, 255, 0): (0, box_size),            # Yellow
                (255, 192, 203): (box_size, -box_size),  # Pink
                (128, 128, 128): (box_size, 0),          # Gray
                (173, 216, 230): (box_size, box_size)    # Light blue
            }
            
            color_key = (dragged_frame_color.red(), dragged_frame_color.green(), dragged_frame_color.blue())
            dragged_offset = color_to_offset.get(color_key, (0, 0))
            
            # Calculate where original should be: dragged_position - dragged_offset = original_position
            original_new_point = (dragged_new_point[0] - dragged_offset[0], 
                                 dragged_new_point[1] - dragged_offset[1])
            
            # Update the original polygon's control point too
            if self.selected_control_point < len(original_polygon['points']):
                original_polygon['points'][self.selected_control_point] = original_new_point
        
        # Define the same offsets used during creation
        offsets = [
            (-box_size, -box_size),  # 1st copy - Red frame
            (-box_size, 0),          # 2nd copy - Blue frame
            (-box_size, box_size),   # 3rd copy - Light green frame
            (0, -box_size),          # 4th copy - Purple frame
            (0, box_size),           # 5th copy - Yellow frame
            (box_size, -box_size),   # 6th copy - Pink frame
            (box_size, 0),           # 7th copy - Gray frame
            (box_size, box_size)     # 8th copy - Light blue frame
        ]
        
        # Update each copy's control point based on the offset from original
        copy_index = 0
        for polygon in group_polygons:
            if polygon is original_polygon:
                continue  # Skip the original polygon
                
            if copy_index < len(offsets) and self.selected_control_point < len(polygon['points']):
                offset_x, offset_y = offsets[copy_index]
                new_x = original_new_point[0] + offset_x
                new_y = original_new_point[1] + offset_y
                polygon['points'][self.selected_control_point] = (new_x, new_y)
                
            copy_index += 1
        
        self.update()  # Refresh display

    def rotate_polygon(self, polygon_index, angle_degrees):
        """Rotate a polygon around its center by the given angle in degrees"""
        if polygon_index < 0 or polygon_index >= len(self.polygons):
            return
            
        polygon = self.polygons[polygon_index]
        points = polygon['points']
        
        if len(points) < 3:
            return
            
        # Calculate polygon center
        center_x = sum(point[0] for point in points) / len(points)
        center_y = sum(point[1] for point in points) / len(points)
        
        # Convert angle to radians
        angle_radians = math.radians(angle_degrees)
        
        # Rotate each point around the center
        rotated_points = []
        for world_x, world_y in points:
            # Translate to origin (relative to center)
            rel_x = world_x - center_x
            rel_y = world_y - center_y
            
            # Apply rotation
            rotated_x = rel_x * math.cos(angle_radians) - rel_y * math.sin(angle_radians)
            rotated_y = rel_x * math.sin(angle_radians) + rel_y * math.cos(angle_radians)
            
            # Translate back
            final_x = rotated_x + center_x
            final_y = rotated_y + center_y
            
            rotated_points.append((final_x, final_y))
        
        # Update the polygon with rotated points
        polygon['points'] = rotated_points

    def rotate_polygon_group(self, polygon_index, angle_degrees):
        """Rotate a polygon and all its copies if duplicate mode is enabled"""
        if polygon_index < 0 or polygon_index >= len(self.polygons):
            return
            
        # Rotate the selected polygon
        self.rotate_polygon(polygon_index, angle_degrees)
        
        # If duplicate mode is enabled, rotate all polygons in the same group
        if self.duplicate_mode:
            selected_polygon = self.polygons[polygon_index]
            group_id = selected_polygon.get('group_id')
            
            if group_id is not None:
                # Find all polygons with the same group ID and rotate them
                for i, polygon in enumerate(self.polygons):
                    if i != polygon_index and polygon.get('group_id') == group_id:
                        self.rotate_polygon(i, angle_degrees)

    def move_polygon(self, polygon_index, delta_x, delta_y):
        """Move a polygon by the given offset"""
        if polygon_index < 0 or polygon_index >= len(self.polygons):
            return
            
        polygon = self.polygons[polygon_index]
        points = polygon['points']
        
        # Move each point by the delta
        moved_points = []
        for world_x, world_y in points:
            moved_points.append((world_x + delta_x, world_y + delta_y))
        
        # Update the polygon with moved points
        polygon['points'] = moved_points

    def move_polygon_group(self, polygon_index, delta_x, delta_y):
        """Move a polygon and all its copies if duplicate mode is enabled"""
        if polygon_index < 0 or polygon_index >= len(self.polygons):
            return
            
        # Move the selected polygon
        self.move_polygon(polygon_index, delta_x, delta_y)
        
        # If duplicate mode is enabled, move all polygons in the same group
        if self.duplicate_mode:
            selected_polygon = self.polygons[polygon_index]
            group_id = selected_polygon.get('group_id')
            
            if group_id is not None:
                # Find all polygons with the same group ID and move them
                for i, polygon in enumerate(self.polygons):
                    if i != polygon_index and polygon.get('group_id') == group_id:
                        self.move_polygon(i, delta_x, delta_y)

    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming or polygon rotation"""
        # If a polygon is selected, use wheel for rotation instead of zoom
        if self.selected_polygon_index >= 0:
            # Calculate rotation angle based on wheel direction
            wheel_delta = event.angleDelta().y()
            rotation_angle = 2 if wheel_delta > 0 else -2  # 5 degrees per wheel step
            
            # Rotate the selected polygon (and its copies if duplicate mode is on)
            self.rotate_polygon_group(self.selected_polygon_index, rotation_angle)
            
            self.update()  # Refresh display
            return
        
        # No polygon selected - proceed with normal zoom behavior
        # Get mouse position before zoom
        mouse_pos = event.pos()
        old_world_x, old_world_y = self.screen_to_world(mouse_pos.x(), mouse_pos.y())
        
        # Update zoom factor
        zoom_in = event.angleDelta().y() > 0
        zoom_factor = 1.25 if zoom_in else 0.8
        
        # Limit zoom range
        new_zoom = self.zoom_factor * zoom_factor
        if 0.001 <= new_zoom <= 1000.0:
            self.zoom_factor = new_zoom
            
            # Adjust pan offset to keep mouse position fixed
            new_screen_x, new_screen_y = self.world_to_screen(old_world_x, old_world_y)
            self.pan_offset_x += mouse_pos.x() - new_screen_x
            self.pan_offset_y += mouse_pos.y() - new_screen_y
            
            self.update()
    
    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key_Escape:
            # Escape key exits all modes except duplicate mode
            modes_changed = False
            
            if self.polygon_mode:
                self.polygon_mode = False
                self.polygon_points = []  # Clear any in-progress polygon
                self.setCursor(Qt.ArrowCursor)
                self.cursor_timer.stop()
                modes_changed = True
                
            if self.eraser_mode:
                self.set_eraser_mode(False)
                modes_changed = True
                
            if self.line_mode:
                self.set_line_mode(False)
                # Clear any in-progress line drawing
                self.is_drawing_line = False
                self.current_line_end = None
                self.line_start_point = None
                self.line_points = []
                modes_changed = True
                
            if modes_changed:
                self.update()
                
                # Find and update the checkboxes - look for them in the widget hierarchy
                parent = self.parent()
                while parent:
                    # Look for SidePanel widgets in the parent's children
                    for child in parent.findChildren(QCheckBox):
                        if child.text() == "Polygon" and hasattr(child, 'setChecked'):
                            # Block signals to prevent triggering toggle_polygon_mode again
                            child.blockSignals(True)
                            child.setChecked(False)
                            child.blockSignals(False)
                        elif child.text() == "Eraser Mode" and hasattr(child, 'setChecked'):
                            child.blockSignals(True)
                            child.setChecked(False)
                            child.blockSignals(False)
                        elif child.text() == "Line" and hasattr(child, 'setChecked'):
                            child.blockSignals(True)
                            child.setChecked(False)
                            child.blockSignals(False)
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
            if self.selected_polygon_index >= 0:
                self.delete_selected_polygon()
        else:
            super().keyPressEvent(event)
    
    def delete_selected_polygon(self):
        """Delete the currently selected polygon and optionally all polygons in its group"""
        if self.selected_polygon_index < 0 or self.selected_polygon_index >= len(self.polygons):
            return
        
        # Save state before deleting
        self.save_state()
        
        # Get the selected polygon and its group ID
        selected_polygon = self.polygons[self.selected_polygon_index]
        group_id = selected_polygon.get('group_id')
        
        # If duplicate mode is enabled and polygon has a group ID, remove all polygons with the same group ID
        if self.duplicate_mode and group_id is not None:
            # Count how many polygons will be removed and track indices
            polygons_to_remove = []
            for i, p in enumerate(self.polygons):
                if p.get('group_id') == group_id:
                    polygons_to_remove.append(i)
            
            # Remove polygons in reverse order to avoid index shifting issues
            for i in reversed(sorted(polygons_to_remove)):
                if i < len(self.polygons):
                    self.polygons.pop(i)
        else:
            # If duplicate mode is off or no group ID, just remove the single polygon
            self.polygons.pop(self.selected_polygon_index)
        
        # Clear overlap data since polygon indices may have changed
        self.overlap_data = []
        self.showing_overlaps = False
        
        # Reset selection to avoid invalid index
        self.selected_polygon_index = -1
        self.selected_control_point = -1
        self.update()
    
    def erase_polygon_at_point(self, world_x, world_y):
        """Erase the polygon or polygon group at the given point"""
        # Find polygon at point
        for i, polygon_data in enumerate(self.polygons):
            points = polygon_data['points']
            if self.point_in_polygon(world_x, world_y, points):
                # Save state before erasing
                self.save_state()
                
                # Get group ID of the polygon to erase
                group_id = polygon_data.get('group_id')
                
                # Only apply group behavior if duplicate mode is currently enabled
                if self.duplicate_mode and group_id is not None:
                    # Remove all polygons with the same group ID using proper indexing
                    polygons_to_remove = []
                    for idx, p in enumerate(self.polygons):
                        if p.get('group_id') == group_id:
                            polygons_to_remove.append(idx)
                    
                    # Remove polygons in reverse order to avoid index shifting issues
                    for idx in reversed(sorted(polygons_to_remove)):
                        if idx < len(self.polygons):
                            self.polygons.pop(idx)
                else:
                    # If duplicate mode is off or no group ID, just remove the single polygon
                    if i < len(self.polygons):
                        self.polygons.pop(i)
                
                # Clear overlap data since polygon indices may have changed
                self.overlap_data = []
                self.showing_overlaps = False
                
                # Reset any selected polygon index to avoid corruption
                if hasattr(self, 'selected_polygon_index'):
                    self.selected_polygon_index = -1
                if hasattr(self, 'selected_control_point'):
                    self.selected_control_point = -1
                
                self.update()
                return True  # Successfully erased polygon(s)
        
        return False  # No polygon found at point
    
    def select_polygon_at_point(self, world_x, world_y):
        """Select a polygon at the given world coordinates"""
        self.selected_polygon_index = -1
        self.selected_polygon_indices = []
        
        # Find all polygons that contain the point
        candidates = []
        for i in range(len(self.polygons)):
            polygon_data = self.polygons[i]
            points = polygon_data['points']
            
            if self.point_in_polygon(world_x, world_y, points):
                # Calculate polygon area to prefer smaller polygons (more precise selection)
                area = self.calculate_polygon_area(points)
                candidates.append((i, area))
        
        if candidates:
            # Sort by area (smallest first) and then by index (most recent first)
            candidates.sort(key=lambda x: (x[1], -x[0]))
            self.selected_polygon_index = candidates[0][0]
            print(f"DEBUG: Selected polygon {self.selected_polygon_index} (area={candidates[0][1]:.2f}) from {len(candidates)} candidates")
        
        self.update()
    
    def calculate_polygon_area(self, points):
        """Calculate the area of a polygon using the shoelace formula"""
        if len(points) < 3:
            return 0
        
        area = 0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        return abs(area) / 2
    
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
    
    def paintEvent(self, event):
        """Paint the canvas"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Fill canvas with white background
        painter.fillRect(self.rect(), self.background_color)
        
        # Apply zoom and pan transformation
        painter.translate(self.pan_offset_x, self.pan_offset_y)
        painter.scale(self.zoom_factor, self.zoom_factor)
        
        # Draw background image if available and enabled
        if self.background_image and not self.background_image.isNull() and self.show_image:
            # Draw image at original size with offset, transformations will handle zoom/pan
            painter.drawPixmap(int(self.image_offset_x), int(self.image_offset_y), self.background_image)
        
        # Reset transformation for UI elements
        painter.resetTransform()
        
        # Draw completed polygons (convert world coordinates to screen)
        for i, polygon_data in enumerate(self.polygons):
            points = polygon_data['points']
            color = polygon_data['color']
            frame_color = polygon_data.get('frame_color', QColor(0, 0, 0))  # Default to black if no frame_color
            
            if len(points) >= 3:
                # Convert world coordinates to screen coordinates
                screen_points = []
                for world_x, world_y in points:
                    screen_x, screen_y = self.world_to_screen(world_x, world_y)
                    screen_points.append(QPoint(int(screen_x), int(screen_y)))
                
                qpolygon = QPolygon(screen_points)
                
                # Highlight selected polygon
                if i == self.selected_polygon_index:
                    # Draw thicker red border for selected polygon
                    painter.setPen(QPen(QColor(255, 0, 0), max(0.5, self.edge_width * self.zoom_factor)))  # Red thick border
                else:
                    # Use the polygon's frame color
                    painter.setPen(QPen(frame_color, max(0.5, self.edge_width * self.zoom_factor)))  # Use frame color with configurable thickness
                
                painter.setBrush(QBrush(color))
                painter.drawPolygon(qpolygon)
        
        # Draw overlap visualization if enabled
        if self.showing_overlaps:
            self.draw_overlap_visualization(painter)
        
        # Draw control points for the selected polygon
        if self.selected_polygon_index >= 0:
            self.draw_control_points(painter)
        
        # Draw grid if enabled
        if self.show_grid:
            self.draw_grid(painter)
        
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
        
        # Draw current line being drawn in line mode
        if self.line_mode and self.is_drawing_line and len(self.line_points) > 1:
            # Draw the original line in light red
            painter.setPen(QPen(QColor(255, 150, 150), 1))  # Light red line
            for i in range(len(self.line_points) - 1):
                start_x, start_y = self.world_to_screen(self.line_points[i][0], self.line_points[i][1])
                end_x, end_y = self.world_to_screen(self.line_points[i + 1][0], self.line_points[i + 1][1])
                painter.drawLine(int(start_x), int(start_y), int(end_x), int(end_y))
            
            # Draw the smooth spline in green for preview
            if len(self.line_points) >= 3:  # Need at least 3 points for spline
                smooth_points, _ = self.create_smooth_spline(self.line_points)
                painter.setPen(QPen(QColor(0, 255, 0), 2))  # Green smooth line
                for i in range(len(smooth_points) - 1):
                    start_x, start_y = self.world_to_screen(smooth_points[i][0], smooth_points[i][1])
                    end_x, end_y = self.world_to_screen(smooth_points[i + 1][0], smooth_points[i + 1][1])
                    painter.drawLine(int(start_x), int(start_y), int(end_x), int(end_y))
    
    def draw_control_points(self, painter):
        """Draw control points for the selected polygon"""
        if self.selected_polygon_index < 0 or self.selected_polygon_index >= len(self.polygons):
            return
        
        # Draw control points for the selected polygon
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
    
    def draw_overlap_visualization(self, painter):
        """Draw overlap visualization with color-coded polygons"""
        # First, color the overlapping polygons
        for poly1_idx, poly2_idx, overlap_points in self.overlap_data:
            # Check if indices are still valid
            if poly1_idx >= len(self.polygons) or poly2_idx >= len(self.polygons):
                continue  # Skip invalid indices
                
            # Color older polygon (lower index) in green
            older_idx = min(poly1_idx, poly2_idx)
            newer_idx = max(poly1_idx, poly2_idx)
            
            # Draw older polygon in semi-transparent green
            older_polygon = self.polygons[older_idx]
            points = older_polygon['points']
            if len(points) >= 3:
                screen_points = []
                for world_x, world_y in points:
                    screen_x, screen_y = self.world_to_screen(world_x, world_y)
                    screen_points.append(QPoint(int(screen_x), int(screen_y)))
                
                qpolygon = QPolygon(screen_points)
                painter.setPen(QPen(QColor(0, 150, 0), max(1.0, self.edge_width * self.zoom_factor * 1.5)))  # Thick green border
                painter.setBrush(QBrush(QColor(0, 255, 0, 100)))  # Semi-transparent green fill
                painter.drawPolygon(qpolygon)
            
            # Draw newer polygon in semi-transparent red
            newer_polygon = self.polygons[newer_idx]
            points = newer_polygon['points']
            if len(points) >= 3:
                screen_points = []
                for world_x, world_y in points:
                    screen_x, screen_y = self.world_to_screen(world_x, world_y)
                    screen_points.append(QPoint(int(screen_x), int(screen_y)))
                
                qpolygon = QPolygon(screen_points)
                painter.setPen(QPen(QColor(150, 0, 0), max(1.0, self.edge_width * self.zoom_factor * 1.5)))  # Thick red border
                painter.setBrush(QBrush(QColor(255, 0, 0, 100)))  # Semi-transparent red fill
                painter.drawPolygon(qpolygon)
        
        # Then, draw the actual overlap areas in bright yellow
        for poly1_idx, poly2_idx, overlap_points in self.overlap_data:
            if len(overlap_points) >= 3:
                screen_points = []
                for world_x, world_y in overlap_points:
                    screen_x, screen_y = self.world_to_screen(world_x, world_y)
                    screen_points.append(QPoint(int(screen_x), int(screen_y)))
                
                qpolygon = QPolygon(screen_points)
                painter.setPen(QPen(QColor(255, 255, 0), max(2.0, self.edge_width * self.zoom_factor * 2)))  # Thick yellow border
                painter.setBrush(QBrush(QColor(255, 255, 0, 150)))  # Semi-transparent yellow fill
                painter.drawPolygon(qpolygon)

    def draw_grid(self, painter):
        """Draw the 3x3 grid overlay with draggable handle that scales with zoom"""
        
        # Calculate grid position and size in world coordinates
        # grid_size is the size of each individual box/cell
        cell_size_world = self.grid_size
        total_grid_size_world = cell_size_world * 3  # 3 boxes = total grid size
        grid_x_world = self.grid_offset_x
        grid_y_world = self.grid_offset_y
        
        # Convert grid corners to screen coordinates
        grid_x_screen, grid_y_screen = self.world_to_screen(grid_x_world, grid_y_world)
        grid_end_x_screen, grid_end_y_screen = self.world_to_screen(
            grid_x_world + total_grid_size_world, grid_y_world + total_grid_size_world)
        
        # Calculate screen cell size
        cell_size_screen = (grid_end_x_screen - grid_x_screen) / 3
        
        # Draw grid lines
        painter.setPen(QPen(QColor(0, 0, 255), 2))  # Blue grid lines
        
        # Draw 3x3 grid (4 lines in each direction to create 3 boxes)
        for i in range(4):
            # Vertical lines
            x_screen = grid_x_screen + (i * cell_size_screen)
            painter.drawLine(int(x_screen), int(grid_y_screen), 
                           int(x_screen), int(grid_end_y_screen))
            
            # Horizontal lines
            y_screen = grid_y_screen + (i * cell_size_screen)
            painter.drawLine(int(grid_x_screen), int(y_screen), 
                           int(grid_end_x_screen), int(y_screen))
        
        # Draw column numbers (1-3) at the top of each column
        painter.setPen(QPen(QColor(0, 0, 255), 1))
        font = QFont()
        font.setPixelSize(max(12, int(cell_size_screen / 8)))  # Scale font with grid
        painter.setFont(font)
        
        for col in range(3):
            x_center = grid_x_screen + (col + 0.5) * cell_size_screen
            y_pos = grid_y_screen - 10  # Above the grid
            painter.drawText(int(x_center - 5), int(y_pos), str(col + 1))
        
        # Draw row numbers (1-3) at the left of each row
        for row in range(3):
            x_pos = grid_x_screen - 15  # Left of the grid
            y_center = grid_y_screen + (row + 0.5) * cell_size_screen
            painter.drawText(int(x_pos), int(y_center + 5), str(row + 1))
        
        # Draw drag handle (small square at top-left corner of grid)
        handle_size = max(8, int(cell_size_screen / 10))
        handle_x = grid_x_screen - handle_size
        handle_y = grid_y_screen - handle_size
        
        painter.setPen(QPen(QColor(255, 0, 0), 2))  # Red handle
        painter.setBrush(QBrush(QColor(255, 255, 255)))  # White fill
        painter.drawRect(int(handle_x), int(handle_y), handle_size, handle_size)
    
    def is_point_in_grid_drag_handle(self, screen_x, screen_y):
        """Check if a screen point is inside the grid drag handle"""
        if not self.show_grid:
            return False
            
        # Calculate grid position and handle position
        cell_size_world = self.grid_size
        total_grid_size_world = cell_size_world * 3
        grid_x_world = self.grid_offset_x
        grid_y_world = self.grid_offset_y
        
        # Convert to screen coordinates
        grid_x_screen, grid_y_screen = self.world_to_screen(grid_x_world, grid_y_world)
        grid_end_x_screen, grid_end_y_screen = self.world_to_screen(
            grid_x_world + total_grid_size_world, grid_y_world + total_grid_size_world)
        
        cell_size_screen = (grid_end_x_screen - grid_x_screen) / 3
        handle_size = max(8, int(cell_size_screen / 10))
        handle_x = grid_x_screen - handle_size
        handle_y = grid_y_screen - handle_size
        
        # Check if point is within handle rectangle
        return (handle_x <= screen_x <= handle_x + handle_size and
                handle_y <= screen_y <= handle_y + handle_size)

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


# ---------------------------------------------------------------------------
# Color picker sub-widgets
# ---------------------------------------------------------------------------

class _HueBar(QWidget):
    """Thin horizontal rainbow strip for selecting hue."""
    hue_changed  = pyqtSignal(float)   # hue in 0-359
    hover_color  = pyqtSignal(object)  # QColor or None

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
        # cursor line
        x = int(self._hue / 359.0 * (w - 1))
        p.setPen(QPen(QColor(0, 0, 0), 2))
        p.drawLine(x, 0, x, h - 1)
        p.setPen(QPen(QColor(255, 255, 255), 1))
        p.drawLine(max(0, x - 1), 0, max(0, x - 1), h - 1)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._hue = self._hue_from_x(event.x())
            self.hue_changed.emit(self._hue)
            self.update()

    def mouseMoveEvent(self, event):
        hue = self._hue_from_x(event.x())
        self.hover_color.emit(QColor.fromHsv(int(hue), 255, 255))
        if event.buttons() & Qt.LeftButton:
            self._hue = hue
            self.hue_changed.emit(self._hue)
            self.update()

    def leaveEvent(self, event):
        self.hover_color.emit(None)


class _SVSquare(QWidget):
    """Saturation / Value gradient square for the current hue."""
    sv_changed  = pyqtSignal(float, float)  # sat, val each in 0-1
    hover_color = pyqtSignal(object)        # QColor or None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hue = 0
        self._sat = 1.0
        self._val = 1.0
        self.setFixedHeight(130)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)

    def set_hue(self, hue):
        self._hue = int(max(0, min(359, hue)))
        self.update()

    def set_sv(self, sat, val):
        self._sat = float(sat)
        self._val = float(val)
        self.update()

    def _sv_from_pos(self, x, y):
        w, h = max(1, self.width()), max(1, self.height())
        return max(0.0, min(1.0, x / w)), max(0.0, min(1.0, 1.0 - y / h))

    def paintEvent(self, event):
        p = QPainter(self)
        w, h = self.width(), self.height()
        # Horizontal: white → pure hue
        hg = QLinearGradient(0, 0, w, 0)
        hg.setColorAt(0.0, QColor(255, 255, 255))
        hg.setColorAt(1.0, QColor.fromHsv(self._hue, 255, 255))
        p.fillRect(0, 0, w, h, hg)
        # Vertical overlay: transparent → black
        vg = QLinearGradient(0, 0, 0, h)
        vg.setColorAt(0.0, QColor(0, 0, 0, 0))
        vg.setColorAt(1.0, QColor(0, 0, 0, 255))
        p.fillRect(0, 0, w, h, vg)
        # Crosshair circle
        cx = int(self._sat * (w - 1))
        cy = int((1.0 - self._val) * (h - 1))
        p.setPen(QPen(QColor(0, 0, 0), 1))
        p.drawEllipse(cx - 5, cy - 5, 10, 10)
        p.setPen(QPen(QColor(255, 255, 255), 1))
        p.drawEllipse(cx - 4, cy - 4, 8, 8)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._sat, self._val = self._sv_from_pos(event.x(), event.y())
            self.sv_changed.emit(self._sat, self._val)
            self.update()

    def mouseMoveEvent(self, event):
        sat, val = self._sv_from_pos(event.x(), event.y())
        self.hover_color.emit(QColor.fromHsv(self._hue, int(sat * 255), int(val * 255)))
        if event.buttons() & Qt.LeftButton:
            self._sat, self._val = sat, val
            self.sv_changed.emit(self._sat, self._val)
            self.update()

    def leaveEvent(self, event):
        self.hover_color.emit(None)


class ColorPickerWidget(QWidget):
    """HSV color picker: hue bar + SV square + RGB spin boxes."""
    color_changed = pyqtSignal(QColor)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating = False
        self._build_ui()
        self._apply_color(QColor(0, 0, 0))

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._hue_bar = _HueBar()
        self._hue_bar.hue_changed.connect(self._hue_changed)
        layout.addWidget(self._hue_bar)

        self._sv = _SVSquare()
        self._sv.sv_changed.connect(self._sv_changed)
        layout.addWidget(self._sv)

        # R G B spinboxes
        rgb_row = QHBoxLayout()
        rgb_row.setSpacing(4)
        self._spins = []
        for label in ('R', 'G', 'B'):
            col = QVBoxLayout()
            col.setSpacing(1)
            lbl = QLabel(label)
            lbl.setAlignment(Qt.AlignCenter)
            sp = QSpinBox()
            sp.setRange(0, 255)
            sp.setAlignment(Qt.AlignCenter)
            sp.setMinimumWidth(52)
            col.addWidget(lbl)
            col.addWidget(sp)
            rgb_row.addLayout(col)
            self._spins.append(sp)
        layout.addLayout(rgb_row)

        for sp in self._spins:
            sp.valueChanged.connect(self._rgb_changed)

    def get_color(self):
        return QColor(self._spins[0].value(), self._spins[1].value(), self._spins[2].value())

    def _apply_color(self, color):
        """Push a QColor into all sub-widgets without feedback loops."""
        self._updating = True
        h, s, v, _ = color.getHsv()   # h: 0-359 or -1 (achromatic)
        if h >= 0:
            # Chromatic: update hue bar and SV square hue
            self._hue_bar.set_hue(h)
            self._sv.set_hue(h)
        # else: achromatic (gray) — leave the hue bar where it is, only update SV
        self._sv.set_sv(s / 255.0, v / 255.0)
        self._spins[0].setValue(color.red())
        self._spins[1].setValue(color.green())
        self._spins[2].setValue(color.blue())
        self._updating = False

    def _hue_changed(self, hue):
        if self._updating:
            return
        self._sv.set_hue(int(hue))
        color = QColor.fromHsv(int(hue), int(self._sv._sat * 255), int(self._sv._val * 255))
        self._updating = True
        self._spins[0].setValue(color.red())
        self._spins[1].setValue(color.green())
        self._spins[2].setValue(color.blue())
        self._updating = False
        self.color_changed.emit(color)

    def _sv_changed(self, sat, val):
        if self._updating:
            return
        color = QColor.fromHsv(int(self._hue_bar._hue), int(sat * 255), int(val * 255))
        self._updating = True
        self._spins[0].setValue(color.red())
        self._spins[1].setValue(color.green())
        self._spins[2].setValue(color.blue())
        self._updating = False
        self.color_changed.emit(color)

    def _rgb_changed(self):
        if self._updating:
            return
        color = QColor(self._spins[0].value(), self._spins[1].value(), self._spins[2].value())
        self._apply_color(color)
        self.color_changed.emit(color)


class _ClickableColorBox(QLabel):
    """Color box label that emits right_clicked signal on right mouse button press."""
    right_clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.right_clicked.emit()
        else:
            super().mousePressEvent(event)


class _SavedPaletteWidget(QWidget):
    """A 16-slot saved-color strip beneath the picker.

    Left-click an empty slot  → save current picker color into it.
    Left-click a filled slot  → load that color back into the picker.
    Right-click any slot       → arm it; the next picker selection fills it.
    """

    slot_loaded  = pyqtSignal(QColor)   # emitted when user loads a slot
    hover_color  = pyqtSignal(object)   # QColor or None when hovering

    SLOTS = 16
    COLS  = 8
    SWATCH = 22
    GAP    = 2

    def __init__(self, get_current_color_fn, parent=None):
        """
        get_current_color_fn: callable that returns the currently picked QColor.
        """
        super().__init__(parent)
        self._get_color = get_current_color_fn
        self._slots = [None] * self.SLOTS   # None = empty
        self._hover_idx = -1
        self._armed_idx = -1   # slot waiting to be filled by the picker
        rows = math.ceil(self.SLOTS / self.COLS)
        w = self.COLS  * (self.SWATCH + self.GAP) + self.GAP
        h = rows       * (self.SWATCH + self.GAP) + self.GAP
        self.setFixedSize(w, h)
        self.setCursor(Qt.PointingHandCursor)
        self.setMouseTracking(True)
        self.setToolTip(
            'Left-click filled slot: load color\n'
            'Right-click any slot: arm it, then pick a color in the picker to fill it'
        )

    def _idx_from_pos(self, x, y):
        col = (x - self.GAP) // (self.SWATCH + self.GAP)
        row = (y - self.GAP) // (self.SWATCH + self.GAP)
        idx = row * self.COLS + col
        if 0 <= col < self.COLS and 0 <= idx < self.SLOTS:
            return idx
        return -1

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        p.fillRect(self.rect(), QColor(180, 180, 180))
        for i in range(self.SLOTS):
            row = i // self.COLS
            col = i % self.COLS
            x = self.GAP + col * (self.SWATCH + self.GAP)
            y = self.GAP + row * (self.SWATCH + self.GAP)
            color = self._slots[i]
            if color is not None:
                p.fillRect(x, y, self.SWATCH, self.SWATCH, color)
                p.setPen(QPen(QColor(60, 60, 60), 1))
            else:
                # Empty slot: checkerboard hint
                p.fillRect(x, y, self.SWATCH, self.SWATCH, QColor(220, 220, 220))
                p.setPen(QPen(QColor(160, 160, 160), 1))
                mid = self.SWATCH // 2
                p.drawLine(x + mid, y + 2, x + mid, y + self.SWATCH - 3)
                p.drawLine(x + 2, y + mid, x + self.SWATCH - 3, y + mid)
                p.setPen(QPen(QColor(160, 160, 160), 1))
            # Highlight armed slot (orange border)
            if i == self._armed_idx:
                p.setPen(QPen(QColor(255, 140, 0), 2))
                p.drawRect(x + 1, y + 1, self.SWATCH - 3, self.SWATCH - 3)
            # Highlight hovered slot (white border)
            elif i == self._hover_idx:
                p.setPen(QPen(QColor(255, 255, 255), 2))
                p.drawRect(x + 1, y + 1, self.SWATCH - 3, self.SWATCH - 3)
            p.setPen(QPen(QColor(60, 60, 60), 1))
            p.drawRect(x, y, self.SWATCH - 1, self.SWATCH - 1)

    def mouseMoveEvent(self, event):
        idx = self._idx_from_pos(event.x(), event.y())
        if idx != self._hover_idx:
            self._hover_idx = idx
            self.update()
        color = self._slots[idx] if 0 <= idx < self.SLOTS else None
        self.hover_color.emit(color)

    def leaveEvent(self, event):
        self._hover_idx = -1
        self.update()
        self.hover_color.emit(None)

    def fill_armed_slot(self, color):
        """Fill the armed slot with the given color, then disarm."""
        if self._armed_idx < 0:
            return
        self._slots[self._armed_idx] = QColor(color)
        self._armed_idx = -1
        self.update()

    def mousePressEvent(self, event):
        idx = self._idx_from_pos(event.x(), event.y())
        if idx < 0:
            return
        if event.button() == Qt.LeftButton:
            if self._slots[idx] is not None:
                # Load saved color into picker
                self.slot_loaded.emit(QColor(self._slots[idx]))
                self.update()
        elif event.button() == Qt.RightButton:
            # Arm this slot to receive the next picker color
            self._armed_idx = idx
            self.update()


class SidePanel(QFrame):
    """Side panel widget"""
    
    def __init__(self, title, canvas=None):
        super().__init__()
        self.setFrameStyle(QFrame.StyledPanel)
        self.setMinimumWidth(200)
        self.setMaximumWidth(250)
        self.setStyleSheet("background-color: #f0f0f0;")
        self.canvas = canvas
        
        # Create layout for the panel
        layout = QVBoxLayout()
        
        # Add buttons for left panel
        if title == "Left Panel" and canvas:
            # Add overlap detection button
            overlap_button = QPushButton("Show Overlaps")
            overlap_button.clicked.connect(self.canvas.show_overlaps)
            layout.addWidget(overlap_button)
            
            # Add delete green polygons button
            delete_green_button = QPushButton("Delete Green")
            delete_green_button.setToolTip("Delete all green (older) overlapping polygons")
            delete_green_button.clicked.connect(self.canvas.delete_green_polygons)
            layout.addWidget(delete_green_button)
            
            # Add delete red polygons button
            delete_red_button = QPushButton("Delete Red")
            delete_red_button.setToolTip("Delete all red (newer) overlapping polygons")
            delete_red_button.clicked.connect(self.canvas.delete_red_polygons)
            layout.addWidget(delete_red_button)
            
            # Add undo button
            undo_button = QPushButton("Undo")
            undo_button.setToolTip("Undo the last action")
            undo_button.clicked.connect(self.canvas.undo_last_action)
            layout.addWidget(undo_button)
            
            # Add background image movement controls
            bg_movement_label = QLabel('Move Background:')
            layout.addWidget(bg_movement_label)
            
            # Create arrow buttons layout
            arrow_layout = QVBoxLayout()
            
            # Up arrow button
            up_button = ContinuousButton("↑", self.move_background_image, 0, -1)
            up_button.setMaximumSize(40, 30)
            up_button.setToolTip("Move background image up (hold for continuous movement)")
            
            # Middle row with left and right arrows
            middle_layout = QHBoxLayout()
            left_button = ContinuousButton("←", self.move_background_image, -1, 0)
            left_button.setMaximumSize(40, 30)
            left_button.setToolTip("Move background image left (hold for continuous movement)")
            
            right_button = ContinuousButton("→", self.move_background_image, 1, 0)
            right_button.setMaximumSize(40, 30)
            right_button.setToolTip("Move background image right (hold for continuous movement)")
            
            middle_layout.addWidget(left_button)
            middle_layout.addStretch()
            middle_layout.addWidget(right_button)
            
            # Down arrow button
            down_button = ContinuousButton("↓", self.move_background_image, 0, 1)
            down_button.setMaximumSize(40, 30)
            down_button.setToolTip("Move background image down (hold for continuous movement)")
            
            # Add buttons to arrow layout
            arrow_layout.addWidget(up_button, 0, Qt.AlignCenter)
            arrow_layout.addLayout(middle_layout)
            arrow_layout.addWidget(down_button, 0, Qt.AlignCenter)
            
            # Add arrow layout to main layout
            layout.addLayout(arrow_layout)
            
            # Add color palette section
            color_label = QLabel('Color Palette:')
            layout.addWidget(color_label)
            
            # Create color palette with specified colors
            self.selected_color = QColor(0, 0, 0)  # Default to black
            self.create_color_palette(layout)
            
            # Add stretch to push save/load buttons to bottom
            layout.addStretch()
            
            # Add save and load array buttons at the bottom
            save_array_button = QPushButton("Save Array")
            save_array_button.clicked.connect(self.save_array)
            layout.addWidget(save_array_button)
            
            load_array_button = QPushButton("Load Array")
            load_array_button.clicked.connect(self.load_array)
            layout.addWidget(load_array_button)
        
        # Add buttons for right panel
        elif title == "Right Panel" and canvas:
            load_bg_button = QPushButton("Load Background")
            load_bg_button.clicked.connect(self.load_background)
            layout.addWidget(load_bg_button)
            
            # Add polygon checkbox
            self.polygon_checkbox = QCheckBox("Polygon")
            self.polygon_checkbox.toggled.connect(self.on_polygon_toggled)
            layout.addWidget(self.polygon_checkbox)
            
            # Add eraser mode checkbox
            self.eraser_checkbox = QCheckBox("Eraser Mode")
            self.eraser_checkbox.toggled.connect(self.on_eraser_toggled)
            layout.addWidget(self.eraser_checkbox)
            
            # Add duplicate mode checkbox
            self.duplicate_checkbox = QCheckBox("Duplicate")
            self.duplicate_checkbox.toggled.connect(self.on_duplicate_toggled)
            layout.addWidget(self.duplicate_checkbox)
            
            # Add mandala mode checkbox
            self.mandala_checkbox = QCheckBox("Mandala")
            self.mandala_checkbox.toggled.connect(self.on_mandala_toggled)
            layout.addWidget(self.mandala_checkbox)
            
            # Add mandala click mode checkbox
            self.mandala_click_checkbox = QCheckBox("Mandala Click")
            self.mandala_click_checkbox.toggled.connect(self.on_mandala_click_toggled)
            layout.addWidget(self.mandala_click_checkbox)
            
            # Add paint mode checkbox
            self.paint_checkbox = QCheckBox("Paint")
            self.paint_checkbox.toggled.connect(self.on_paint_toggled)
            layout.addWidget(self.paint_checkbox)
            
            # Add random mode checkbox
            self.random_checkbox = QCheckBox("Random")
            self.random_checkbox.toggled.connect(self.on_random_toggled)
            layout.addWidget(self.random_checkbox)
            
            # Add duplicate button
            duplicate_button = QPushButton("Duplicate")
            duplicate_button.clicked.connect(self.duplicate_all_polygons)
            layout.addWidget(duplicate_button)
            
            # Add line mode checkbox
            self.line_checkbox = QCheckBox("Line")
            self.line_checkbox.toggled.connect(self.on_line_toggled)
            layout.addWidget(self.line_checkbox)
            
            # Add smoothing parameter input
            smoothing_label = QLabel('Line Smoothing:')
            layout.addWidget(smoothing_label)
            
            self.smoothing_input = QLineEdit()
            self.smoothing_input.setText('0.05')  # Default value
            self.smoothing_input.setPlaceholderText('Enter smoothing factor (0.01-2.0)')
            self.smoothing_input.textChanged.connect(self.on_smoothing_changed)
            layout.addWidget(self.smoothing_input)
            
            # Add polygon size parameter input
            polygon_size_label = QLabel('Polygon Size (pixels):')
            layout.addWidget(polygon_size_label)
            
            self.polygon_size_input = QLineEdit()
            self.polygon_size_input.setText('15')  # Default value
            self.polygon_size_input.setPlaceholderText('Enter polygon size in pixels')
            self.polygon_size_input.textChanged.connect(self.on_polygon_size_changed)
            layout.addWidget(self.polygon_size_input)
            
            # Parallel lines input
            parallel_lines_label = QLabel('Parallel Lines (each side):')
            layout.addWidget(parallel_lines_label)
            
            self.parallel_lines_input = QLineEdit()
            self.parallel_lines_input.setText('0')  # Default value - no parallel lines
            self.parallel_lines_input.setPlaceholderText('Enter number of parallel lines on each side')
            self.parallel_lines_input.textChanged.connect(self.on_parallel_lines_changed)
            layout.addWidget(self.parallel_lines_input)
            
            # Gap between parallel lines input
            parallel_gap_label = QLabel('Gap Between Lines:')
            layout.addWidget(parallel_gap_label)
            
            self.parallel_gap_input = QLineEdit()
            self.parallel_gap_input.setText('2')  # Default gap value
            self.parallel_gap_input.setPlaceholderText('Enter gap between parallel lines (e.g., 2)')
            self.parallel_gap_input.textChanged.connect(self.on_parallel_gap_changed)
            layout.addWidget(self.parallel_gap_input)
            
            # Gap between polygons input
            gap_label = QLabel('Gap Between Polygons:')
            layout.addWidget(gap_label)
            
            self.gap_input = QLineEdit()
            self.gap_input.setText('2')  # Default gap value
            self.gap_input.setPlaceholderText('Enter gap between polygons (e.g., 2)')
            self.gap_input.textChanged.connect(self.on_gap_changed)
            layout.addWidget(self.gap_input)
            
            # Add show image checkbox
            self.show_image_checkbox = QCheckBox("Show Image")
            self.show_image_checkbox.setChecked(True)  # Checked by default
            self.show_image_checkbox.toggled.connect(self.on_show_image_toggled)
            layout.addWidget(self.show_image_checkbox)
            
            # X scale input
            x_scale_label = QLabel('X Scale (%):')
            layout.addWidget(x_scale_label)
            
            self.x_scale_input = QLineEdit()
            self.x_scale_input.setText('100')
            self.x_scale_input.setPlaceholderText('Enter X scale percentage (e.g., 100)')
            self.x_scale_input.textChanged.connect(self.on_scale_changed)
            layout.addWidget(self.x_scale_input)
            
            # Y scale input
            y_scale_label = QLabel('Y Scale (%):')
            layout.addWidget(y_scale_label)
            
            self.y_scale_input = QLineEdit()
            self.y_scale_input.setText('100')
            self.y_scale_input.setPlaceholderText('Enter Y scale percentage (e.g., 100)')
            self.y_scale_input.textChanged.connect(self.on_scale_changed)
            layout.addWidget(self.y_scale_input)
            
            # Add grid checkbox
            self.grid_checkbox = QCheckBox("Show Grid")
            self.grid_checkbox.setChecked(False)  # Unchecked by default
            self.grid_checkbox.toggled.connect(self.on_grid_toggled)
            layout.addWidget(self.grid_checkbox)
            
            # Grid size input
            grid_size_label = QLabel('Box Size:')
            layout.addWidget(grid_size_label)
            
            self.grid_size_input = QLineEdit()
            self.grid_size_input.setText('300')
            self.grid_size_input.setPlaceholderText('Enter box size (e.g., 300)')
            self.grid_size_input.textChanged.connect(self.on_grid_size_changed)
            layout.addWidget(self.grid_size_input)
            
            # Edge width input
            edge_width_label = QLabel('Edge Width:')
            layout.addWidget(edge_width_label)
            
            self.edge_width_input = QLineEdit()
            self.edge_width_input.setText('1.1')
            self.edge_width_input.setPlaceholderText('Enter edge width (e.g., 1.1)')
            self.edge_width_input.textChanged.connect(self.on_edge_width_changed)
            layout.addWidget(self.edge_width_input)
            
            # Add stretch to push cursor position to bottom (only for right panel)
            layout.addStretch()
            
            # Add cursor position label at the bottom
            self.cursor_position_label = QLabel('Cursor: (0, 0)')
            self.cursor_position_label.setStyleSheet("font-family: monospace; font-size: 10px; color: #666;")
            self.cursor_position_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.cursor_position_label)
        
        # For left panel, stretch was already added before save/load buttons
        
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
            # Ask for target resolution
            width, ok_width = QInputDialog.getInt(
                self,
                "Target Resolution",
                "Enter target width (pixels):",
                value=1000,
                min=1,
                max=100000
            )
            
            if not ok_width:
                return
            
            height, ok_height = QInputDialog.getInt(
                self,
                "Target Resolution",
                "Enter target height (pixels):",
                value=1000,
                min=1,
                max=100000
            )
            
            if not ok_height:
                return
            
            # Load the original image first
            from PyQt5.QtGui import QPixmap
            original_pixmap = QPixmap(file_path)
            if original_pixmap.isNull():
                QMessageBox.warning(self, "Error", "Failed to load image.")
                return
            
            # Resize to target resolution
            resized_pixmap = original_pixmap.scaled(
                width, height,
                Qt.IgnoreAspectRatio,
                Qt.SmoothTransformation
            )
            
            # Save the resized image temporarily and load it
            import tempfile
            import os
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            temp_path = temp_file.name
            temp_file.close()
            
            resized_pixmap.save(temp_path)
            self.canvas.set_background_image(temp_path)
            
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
            
            # Reset scale inputs to 100%
            if hasattr(self, 'x_scale_input'):
                self.x_scale_input.blockSignals(True)  # Prevent triggering scale change
                self.x_scale_input.setText('100')
                self.x_scale_input.blockSignals(False)
                
            if hasattr(self, 'y_scale_input'):
                self.y_scale_input.blockSignals(True)  # Prevent triggering scale change
                self.y_scale_input.setText('100')
                self.y_scale_input.blockSignals(False)
    
    def move_background_image(self, delta_x, delta_y):
        """Move background image by the specified delta in pixels"""
        if not self.canvas or not self.canvas.background_image:
            return
            
        # Move the background image offset
        self.canvas.image_offset_x += delta_x
        self.canvas.image_offset_y += delta_y
        
        # Update the canvas display
        self.canvas.update()
    
    def on_polygon_toggled(self, checked):
        """Handle polygon checkbox toggle"""
        if self.canvas:
            self.canvas.toggle_polygon_mode()
    
    def on_eraser_toggled(self, checked):
        """Handle eraser mode checkbox toggle"""
        if self.canvas:
            self.canvas.set_eraser_mode(checked)
    
    def on_duplicate_toggled(self, checked):
        """Handle duplicate mode checkbox toggle"""
        if self.canvas:
            self.canvas.duplicate_mode = checked
    
    def on_mandala_toggled(self, checked):
        """Handle mandala mode checkbox toggle"""
        if self.canvas:
            self.canvas.mandala_mode = checked
    
    def on_mandala_click_toggled(self, checked):
        """Handle mandala click mode checkbox toggle"""
        if self.canvas:
            self.canvas.mandala_click_mode = checked
            
            # If disabling mandala click mode, remove green fill from clicked polygons
            if not checked:
                self.remove_mandala_click_markings()
    
    def on_paint_toggled(self, checked):
        """Handle paint mode checkbox toggle"""
        if self.canvas:
            self.canvas.paint_mode = checked
    
    def on_random_toggled(self, checked):
        """Handle random mode checkbox toggle"""
        if self.canvas:
            self.canvas.random_mode = checked
    
    def remove_mandala_click_markings(self):
        """Remove green fill from polygons that were marked in mandala click mode"""
        if not self.canvas:
            return
            
        for polygon in self.canvas.polygons:
            if polygon.get('mandala_clicked', False):
                # Restore original transparent fill
                polygon['color'] = QColor(0, 0, 0, 0)  # Transparent fill
                # Remove the mandala clicked marker
                if 'mandala_clicked' in polygon:
                    del polygon['mandala_clicked']
        
        self.canvas.update()  # Refresh display
    
    def create_color_palette(self, layout):
        """Create HSV color picker widget + hover preview + 16-slot saved palette below."""
        from PyQt5.QtWidgets import QHBoxLayout

        self.color_picker = ColorPickerWidget()
        self.color_picker.color_changed.connect(self._on_picker_color_changed)
        layout.addWidget(self.color_picker)

        # Color boxes row: chosen color + hover preview
        boxes_row = QHBoxLayout()
        boxes_row.setSpacing(6)

        # Chosen color box (default white, updated on picker click)
        boxes_row.addWidget(QLabel('Color:'))
        self._chosen_color_box = _ClickableColorBox()
        self._chosen_color_box.setFixedSize(36, 22)
        self._chosen_color_box.setStyleSheet('background-color: rgb(255,255,255); border: 2px solid #555;')
        self._chosen_color_box.setToolTip('Active color — right-click to pick color from canvas')
        self._chosen_color_box.right_clicked.connect(self._start_eyedropper)
        boxes_row.addWidget(self._chosen_color_box)

        # Eyedropper button (explicit, always visible)
        self._eyedropper_btn = QPushButton('⊕')
        self._eyedropper_btn.setFixedSize(22, 22)
        self._eyedropper_btn.setToolTip('Pick color from canvas (then right-click a polygon)')
        self._eyedropper_btn.setCheckable(True)
        self._eyedropper_btn.clicked.connect(self._on_eyedropper_btn_clicked)
        boxes_row.addWidget(self._eyedropper_btn)

        boxes_row.addSpacing(4)

        # Hover preview box
        boxes_row.addWidget(QLabel('Hover:'))
        self._hover_preview = QLabel()
        self._hover_preview.setFixedSize(36, 22)
        self._hover_preview.setStyleSheet('background-color: transparent; border: 1px solid #888;')
        boxes_row.addWidget(self._hover_preview)
        boxes_row.addStretch()
        layout.addLayout(boxes_row)

        # Saved-color palette strip
        self._saved_palette = _SavedPaletteWidget(self.color_picker.get_color)
        self._saved_palette.slot_loaded.connect(self._load_saved_color)
        self._saved_palette.hover_color.connect(self._on_palette_hover)
        self.color_picker._hue_bar.hover_color.connect(self._on_palette_hover)
        self.color_picker._sv.hover_color.connect(self._on_palette_hover)
        self.color_picker.color_changed.connect(self._saved_palette.fill_armed_slot)
        layout.addWidget(self._saved_palette)

        # Replace-color row
        self._replace_from_color = None
        replace_row = QHBoxLayout()
        replace_row.setSpacing(4)
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
        replace_row.addWidget(QLabel('→ current color'))
        self._replace_apply_btn = QPushButton('Apply')
        self._replace_apply_btn.setFixedSize(40, 22)
        self._replace_apply_btn.setToolTip('Replace all polygons with source color → current color')
        self._replace_apply_btn.clicked.connect(self._apply_replace_color)
        replace_row.addWidget(self._replace_apply_btn)
        replace_row.addStretch()
        layout.addLayout(replace_row)

        # Color variance row
        variance_row = QHBoxLayout()
        variance_row.setSpacing(4)
        variance_btn = QPushButton('Vary Colors')
        variance_btn.setToolTip('Randomly shift every polygon color by up to the variance amount')
        variance_btn.clicked.connect(self._apply_color_variance)
        variance_row.addWidget(variance_btn)
        self._variance_slider = QSlider(Qt.Horizontal)
        self._variance_slider.setRange(0, 128)
        self._variance_slider.setValue(20)
        self._variance_slider.setFixedWidth(80)
        self._variance_slider.setToolTip('Max color variance per channel (0–128)')
        variance_row.addWidget(self._variance_slider)
        self._variance_label = QLabel('20')
        self._variance_label.setFixedWidth(24)
        self._variance_slider.valueChanged.connect(lambda v: self._variance_label.setText(str(v)))
        variance_row.addWidget(self._variance_label)
        variance_row.addStretch()
        layout.addLayout(variance_row)

        # Background color button
        bg_row = QHBoxLayout()
        bg_row.setSpacing(4)
        bg_row.addWidget(QLabel('Background:'))
        self._bg_color_box = QLabel()
        self._bg_color_box.setFixedSize(36, 22)
        self._bg_color_box.setStyleSheet('background-color: rgb(255,255,255); border: 2px solid #555;')
        bg_row.addWidget(self._bg_color_box)
        bg_btn = QPushButton('Change')
        bg_btn.setFixedSize(55, 22)
        bg_btn.setToolTip('Change the canvas background color')
        bg_btn.clicked.connect(self._change_background_color)
        bg_row.addWidget(bg_btn)
        bg_row.addStretch()
        layout.addLayout(bg_row)

    def _change_background_color(self):
        """Open a color dialog to change the canvas background color."""
        if not self.canvas:
            return
        color = QColorDialog.getColor(self.canvas.background_color, self, 'Choose Background Color')
        if color.isValid():
            self.canvas.background_color = color
            r, g, b = color.red(), color.green(), color.blue()
            self._bg_color_box.setStyleSheet(f'background-color: rgb({r},{g},{b}); border: 2px solid #555;')
            self.canvas.update()

    def _on_replace_sample_btn_clicked(self, checked):
        """Toggle replace-source eyedropper via its button."""
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
        """Called by Canvas after replace-source eyedropper samples a polygon."""
        self._replace_from_color = color
        r, g, b = color.red(), color.green(), color.blue()
        self._replace_from_box.setStyleSheet(
            f'background-color: rgb({r},{g},{b}); border: 2px solid #555;'
        )
        self._replace_sample_btn.setChecked(False)
        self._replace_sample_btn.setStyleSheet('')

    def _apply_replace_color(self):
        """Replace all polygons matching the source color with the current selected color."""
        if self._replace_from_color is None or not self.canvas:
            return
        src = self._replace_from_color
        dst = self.selected_color
        count = 0
        self.canvas.save_state()
        for polygon in self.canvas.polygons:
            c = polygon.get('color')
            if c is not None and c.red() == src.red() and c.green() == src.green() and c.blue() == src.blue():
                polygon['color'] = QColor(dst)
                count += 1
        self.canvas.update()
        print(f'[replace] replaced {count} polygons')

    def _apply_color_variance(self):
        """Randomly shift each polygon's color by up to variance per channel."""
        if not self.canvas or not self.canvas.polygons:
            return
        variance = self._variance_slider.value()
        self.canvas.save_state()
        for polygon in self.canvas.polygons:
            c = polygon.get('color')
            if c is None or c.alpha() == 0:
                continue  # skip transparent polygons
            if c.red() == c.green() == c.blue():
                # Grayscale — shift all channels by the same delta to stay grayscale
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
        """Activate eyedropper mode on the canvas."""
        if self.canvas:
            self.canvas.eyedropper_mode = True
            self.canvas.setCursor(Qt.CrossCursor)
        self._eyedropper_btn.setChecked(True)
        self._eyedropper_btn.setStyleSheet('background-color: #f90; border: 2px solid #c60;')

    def _on_eyedropper_btn_clicked(self, checked):
        """Toggle eyedropper mode via the button."""
        if checked:
            self._start_eyedropper()
        else:
            if self.canvas:
                self.canvas.eyedropper_mode = False
                self.canvas.setCursor(Qt.ArrowCursor)
            self._eyedropper_btn.setStyleSheet('')

    def receive_eyedropper_color(self, color):
        """Called by Canvas after eyedropper samples a polygon color."""
        self.color_picker._apply_color(color)
        self.selected_color = color
        r, g, b = color.red(), color.green(), color.blue()
        self._chosen_color_box.setStyleSheet(
            f'background-color: rgb({r},{g},{b}); border: 2px solid #555;'
        )
        # If a palette slot is armed, fill it with the sampled color
        self._saved_palette.fill_armed_slot(color)
        self._eyedropper_btn.setChecked(False)
        self._eyedropper_btn.setStyleSheet('')

    def _on_palette_hover(self, color):
        """Update the hover preview box."""
        if color is None:
            self._hover_preview.setStyleSheet('background-color: transparent; border: 1px solid #888;')
        else:
            r, g, b = color.red(), color.green(), color.blue()
            self._hover_preview.setStyleSheet(
                f'background-color: rgb({r},{g},{b}); border: 1px solid #888;'
            )

    def _load_saved_color(self, color):
        """Load a saved slot color into the picker and update selected_color."""
        self.color_picker._apply_color(color)
        self.selected_color = color
        r, g, b = color.red(), color.green(), color.blue()
        self._chosen_color_box.setStyleSheet(
            f'background-color: rgb({r},{g},{b}); border: 2px solid #555;'
        )

    def _on_picker_color_changed(self, color):
        """Update selected_color and chosen color box when the picker changes."""
        self.selected_color = color
        r, g, b = color.red(), color.green(), color.blue()
        self._chosen_color_box.setStyleSheet(
            f'background-color: rgb({r},{g},{b}); border: 2px solid #555;'
        )
    
    def duplicate_all_polygons(self):
        """Create 8 copies of all existing polygons using duplicate offsets"""
        if not self.canvas:
            return
        
        # Get all existing polygons
        original_polygons = self.canvas.polygons[:]  # Make a copy of the current polygons
        
        if not original_polygons:
            return  # Nothing to duplicate
        
        # Use the same box_size as in duplicate mode
        box_size = self.canvas.grid_size
        
        # Define the 8 offsets and corresponding frame colors (same as in duplicate mode)
        offsets_and_colors = [
            ((-box_size, -box_size), QColor(255, 0, 0, 255)),    # 1st copy - Red frame
            ((-box_size, 0), QColor(0, 0, 255, 255)),            # 2nd copy - Blue frame
            ((-box_size, box_size), QColor(144, 238, 144, 255)), # 3rd copy - Light green frame
            ((0, -box_size), QColor(128, 0, 128, 255)),          # 4th copy - Purple frame
            ((0, box_size), QColor(255, 255, 0, 255)),           # 5th copy - Yellow frame
            ((box_size, -box_size), QColor(255, 192, 203, 255)), # 6th copy - Pink frame
            ((box_size, 0), QColor(128, 128, 128, 255)),         # 7th copy - Gray frame
            ((box_size, box_size), QColor(173, 216, 230, 255))   # 8th copy - Light blue frame
        ]
        
        # Create duplicates for each original polygon
        for original_polygon in original_polygons:
            for (offset_x, offset_y), frame_color in offsets_and_colors:
                duplicate_points = []
                for point in original_polygon['points']:
                    new_x = point[0] + offset_x
                    new_y = point[1] + offset_y
                    duplicate_points.append((new_x, new_y))
                
                duplicate_polygon = {
                    'points': duplicate_points,
                    'color': QColor(0, 0, 0, 0),  # Transparent fill
                    'frame_color': frame_color,   # Colored frame
                    'group_id': None  # No group ID for manually duplicated polygons
                }
                
                self.canvas.polygons.append(duplicate_polygon)
        
        # Update the display
        self.canvas.update()
    
    def on_line_toggled(self, checked):
        """Handle line mode checkbox toggle"""
        if self.canvas:
            self.canvas.set_line_mode(checked)
    
    def on_smoothing_changed(self, text):
        """Handle smoothing parameter text change"""
        if self.canvas:
            try:
                smoothing_value = float(text)
                # Clamp the value between reasonable bounds
                smoothing_value = max(0.0, min(5.0, smoothing_value))
                self.canvas.line_smoothing_factor = smoothing_value
            except ValueError:
                # If invalid input, keep the current value
                pass
    
    def on_polygon_size_changed(self, text):
        """Handle polygon size parameter text change"""
        if self.canvas:
            try:
                size_value = float(text)
                # Clamp the value between reasonable bounds (1 to 100 pixels)
                size_value = max(1.0, min(100.0, size_value))
                self.canvas.line_polygon_size = size_value
            except ValueError:
                # If invalid input, keep the current value
                pass
    
    def on_parallel_lines_changed(self, text):
        """Handle parallel lines parameter text change"""
        if self.canvas:
            try:
                lines_value = int(text)
                # Clamp the value between reasonable bounds (0 to 10 parallel lines per side)
                lines_value = max(0, min(10, lines_value))
                self.canvas.num_parallel_lines = lines_value
            except ValueError:
                # If invalid input, keep the current value
                pass
    
    def on_parallel_gap_changed(self, text):
        """Handle gap between parallel lines parameter text change"""
        if self.canvas:
            try:
                gap_value = float(text)
                # Clamp the value between reasonable bounds (0 to 100 pixels)
                gap_value = max(0.0, min(100.0, gap_value))
                self.canvas.parallel_line_gap = gap_value
            except ValueError:
                # If invalid input, keep the current value
                pass
    
    def on_gap_changed(self, text):
        """Handle gap parameter text change"""
        if self.canvas:
            try:
                gap_value = float(text)
                # Clamp the value between reasonable bounds (0 to 50 pixels)
                gap_value = max(0.0, min(50.0, gap_value))
                self.canvas.line_polygon_gap = gap_value
            except ValueError:
                # If invalid input, keep the current value
                pass
    
    def on_show_image_toggled(self, checked):
        """Handle show image checkbox toggle"""
        if self.canvas:
            self.canvas.set_image_visible(checked)
    
    def on_grid_toggled(self, checked):
        """Handle grid checkbox toggle"""
        if self.canvas:
            self.canvas.show_grid = checked
            self.canvas.update()
    
    def on_grid_size_changed(self):
        """Handle grid size changes"""
        try:
            text = self.grid_size_input.text().strip()
            if not text:
                return  # Don't change anything for empty input
                
            grid_size = int(text)
            grid_size = max(10, min(2000, grid_size))  # Clamp between 10 and 2000
            
            if self.canvas:
                self.canvas.grid_size = grid_size
                # Re-center image on grid if there's an image loaded
                self.canvas.center_image_on_grid()
                self.canvas.update()
        except ValueError:
            # Invalid input, ignore
            pass
    
    def on_edge_width_changed(self):
        """Handle edge width changes"""
        try:
            text = self.edge_width_input.text().strip()
            if not text:
                return  # Don't change anything for empty input
                
            width = float(text)
            width = max(0.1, min(10.0, width))  # Clamp between 0.1 and 10.0
            
            if self.canvas:
                self.canvas.edge_width = width
                self.canvas.update()
        except ValueError:
            # Invalid input, ignore
            pass
    
    def on_scale_changed(self):
        """Handle X and Y scale changes"""
        try:
            x_text = self.x_scale_input.text().strip()
            y_text = self.y_scale_input.text().strip()
            
            if not x_text or not y_text:
                return  # Don't change anything for empty input
                
            x_scale_percentage = float(x_text)
            y_scale_percentage = float(y_text)
            
            # Clamp between 1% and 1000%
            x_scale_percentage = max(1, min(1000, x_scale_percentage))
            y_scale_percentage = max(1, min(1000, y_scale_percentage))
            
            if self.canvas:
                self.canvas.scale_background_image(x_scale_percentage / 100.0, y_scale_percentage / 100.0)
        except ValueError:
            # Invalid input, ignore
            pass
    
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
                
                # Write header with frame color support, group ID, and image transform parameters
                writer.writerow(['polygon_id', 'coordinates', 'color_r', 'color_g', 'color_b', 'color_a', 
                               'frame_r', 'frame_g', 'frame_b', 'frame_a', 'group_id'])
                
                # Write each polygon
                for i, polygon_data in enumerate(self.canvas.polygons):
                    points = polygon_data['points']
                    color = polygon_data['color']
                    frame_color = polygon_data.get('frame_color', QColor(0, 0, 0, 255))  # Default to black
                    group_id = polygon_data.get('group_id', '')  # Get group ID if available
                    
                    # Convert points to be relative to default grid position (0,0)
                    # Subtract current grid offset to make coordinates relative to origin
                    adjusted_points = []
                    for x, y in points:
                        adjusted_x = x - self.canvas.grid_offset_x
                        adjusted_y = y - self.canvas.grid_offset_y
                        adjusted_points.append([float(adjusted_x), float(adjusted_y)])
                    
                    # Convert points to JSON string format
                    coords_json = json.dumps(adjusted_points)
                    
                    # Extract RGBA values (convert from QColor to 0-1 range)
                    r = color.red() / 255.0
                    g = color.green() / 255.0
                    b = color.blue() / 255.0
                    a = color.alpha() / 255.0
                    
                    # Extract frame color RGBA values
                    fr = frame_color.red() / 255.0
                    fg = frame_color.green() / 255.0
                    fb = frame_color.blue() / 255.0
                    fa = frame_color.alpha() / 255.0
                    
                    # Write row with group ID
                    writer.writerow([i, coords_json, r, g, b, a, fr, fg, fb, fa, group_id])
            
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
        
        # Ask for scale factor
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
                        # Remove quotes and parse as JSON
                        coords_str = coords_str.strip('"\'')
                        
                        try:
                            coord_list = json.loads(coords_str)
                            # Convert loaded coordinates to current grid position and apply scale
                            # Add current grid offset to make coordinates match current grid
                            points = []
                            for point in coord_list:
                                scaled_x = float(point[0]) * scale_factor
                                scaled_y = float(point[1]) * scale_factor
                                adjusted_x = scaled_x + self.canvas.grid_offset_x
                                adjusted_y = scaled_y + self.canvas.grid_offset_y
                                points.append((adjusted_x, adjusted_y))
                        except:
                            # Fallback to ast parsing for backward compatibility
                            import ast
                            coord_list = ast.literal_eval(coords_str)
                            # Convert loaded coordinates to current grid position and apply scale
                            points = []
                            for point in coord_list:
                                scaled_x = float(point[0]) * scale_factor
                                scaled_y = float(point[1]) * scale_factor
                                adjusted_x = scaled_x + self.canvas.grid_offset_x
                                adjusted_y = scaled_y + self.canvas.grid_offset_y
                                points.append((adjusted_x, adjusted_y))
                        
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
                        
                        # Parse frame color if available
                        if 'frame_r' in row and 'frame_g' in row and 'frame_b' in row:
                            fr = float(row['frame_r'])
                            fg = float(row['frame_g'])
                            fb = float(row['frame_b'])
                            
                            # Check for frame alpha channel
                            if 'frame_a' in row:
                                fa = float(row['frame_a'])
                                fa = int(fa * 255) if fa <= 1.0 else int(fa)
                            else:
                                fa = 255  # Default to fully opaque
                            
                            # Convert from 0-1 range to 0-255
                            fr = int(fr * 255) if fr <= 1.0 else int(fr)
                            fg = int(fg * 255) if fg <= 1.0 else int(fg)
                            fb = int(fb * 255) if fb <= 1.0 else int(fb)
                            
                            frame_color = QColor(fr, fg, fb, fa)
                        else:
                            # Default frame color if no frame color data
                            frame_color = QColor(0, 0, 0, 255)  # Black frame
                        
                        # Parse group ID if available
                        group_id = None
                        if 'group_id' in row and row['group_id']:
                            try:
                                group_id = int(row['group_id'])
                            except:
                                group_id = None
                        
                        # Create polygon data structure
                        polygon_data = {
                            'points': points,
                            'color': color,
                            'frame_color': frame_color,
                            'group_id': group_id
                        }
                        polygons.append(polygon_data)
                        
                    except Exception as e:
                        print(f"Error parsing row {row_num}: {e}")
                        continue
            
            if polygons:
                # Clear existing polygons and load new ones
                self.canvas.polygons = polygons
                
                # Update next_group_id to avoid conflicts with loaded polygons
                max_group_id = 0
                for polygon in polygons:
                    group_id = polygon.get('group_id')
                    if group_id is not None and isinstance(group_id, int):
                        max_group_id = max(max_group_id, group_id)
                
                self.canvas.next_group_id = max_group_id + 1
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

    def reset_grid_to_origin(self):
        """Reset grid position to origin (0,0) - useful for aligning loaded arrays"""
        if not self.canvas:
            return
            
        # Get current grid offset
        current_offset_x = self.canvas.grid_offset_x
        current_offset_y = self.canvas.grid_offset_y
        
        if abs(current_offset_x) < 1e-6 and abs(current_offset_y) < 1e-6:
            QMessageBox.information(self, "Info", "Grid is already at origin (0,0)")
            return
        
        # Ask user if they want to move polygons with the grid
        reply = QMessageBox.question(
            self, 
            "Reset Grid", 
            f"Current grid offset: ({current_offset_x:.1f}, {current_offset_y:.1f})\n\n"
            "Do you want to move all polygons to maintain their relative positions to the grid?\n\n"
            "Yes: Move polygons with grid (recommended for loaded arrays)\n"
            "No: Only move grid, leave polygons in place",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Cancel:
            return
        elif reply == QMessageBox.Yes:
            # Move polygons to maintain relative position to grid
            offset_x = -current_offset_x
            offset_y = -current_offset_y
            
            for polygon in self.canvas.polygons:
                adjusted_points = []
                for x, y in polygon['points']:
                    new_x = x + offset_x
                    new_y = y + offset_y
                    adjusted_points.append((new_x, new_y))
                polygon['points'] = adjusted_points
        
        # Reset grid to origin
        self.canvas.grid_offset_x = 0
        self.canvas.grid_offset_y = 0
        self.canvas.update()
        
        QMessageBox.information(self, "Success", "Grid reset to origin (0,0)")

    def update_cursor_position(self, x, y):
        """Update the cursor position label with world coordinates"""
        if hasattr(self, 'cursor_position_label'):
            self.cursor_position_label.setText(f'Cursor: ({x:.1f}, {y:.1f})')


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
        
        # Create central canvas
        self.canvas = Canvas()
        
        # Create left panel (with reference to canvas for overlap functions)
        self.left_panel = SidePanel("Left Panel", self.canvas)
        main_layout.addWidget(self.left_panel)
        
        main_layout.addWidget(self.canvas, 1)  # Give canvas stretch factor of 1
        
        # Create right panel (with reference to canvas for background loading)
        self.right_panel = SidePanel("Right Panel", self.canvas)
        self.canvas.right_panel = self.right_panel  # Store reference for cursor position updates
        self.canvas.left_panel = self.left_panel   # Store reference for color palette access
        main_layout.addWidget(self.right_panel)
        
        central_widget.setLayout(main_layout)


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Create and show the main window
    window = MandalaMosaicWindow()
    window.showMaximized()  # Maximize window but keep title bar
    
    # Start the event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
