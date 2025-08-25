#!/usr/bin/env python3
"""
Watch a folder of 8x8 PNG images and display them as an 8x8 tiled grid.
Each tile randomly selects a texture from the folder and applies a random rotation.
The display updates automatically when files in the folder change.
"""

import sys
import os
import time
import random
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class MultiTextureTiler:
    def __init__(self, folder_path):
        self.folder_path = Path(folder_path)
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.im_display = None
        self.files_modified = True
        self.textures_cache = {}
        self.last_file_list = set()
        
        # Validate folder exists
        if not self.folder_path.exists() or not self.folder_path.is_dir():
            raise FileNotFoundError(f"Folder not found or not a directory: {folder_path}")
        
        # Initial display
        self.update_display()
        
    def load_textures(self):
        """Load all 8x8 PNG textures from the folder."""
        textures = {}
        png_files = list(self.folder_path.glob('*.png'))
        
        if not png_files:
            print(f"Warning: No PNG files found in {self.folder_path}")
            return textures
        
        for png_path in png_files:
            try:
                # Load the image
                img = Image.open(png_path)
                
                # Convert to RGBA if not already (to handle transparency)
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                # Ensure it's 8x8
                if img.size != (8, 8):
                    print(f"Warning: {png_path.name} is {img.size}, resizing to 8x8")
                    img = img.resize((8, 8), Image.NEAREST)
                
                # Convert to numpy array and store
                textures[png_path.name] = np.array(img)
                
            except Exception as e:
                print(f"Error loading {png_path.name}: {e}")
        
        return textures
    
    def create_tiled_image(self):
        """Create a 64x64 tiled image with random texture selection and rotation."""
        # Load all textures
        textures = self.load_textures()
        
        if not textures:
            # Create a default checkered pattern if no textures available
            default = np.zeros((8, 8, 4), dtype=np.uint8)
            default[::2, ::2] = [255, 0, 255, 255]  # Magenta
            default[1::2, 1::2] = [255, 0, 255, 255]
            textures = {"default": default}
        
        # Update cache
        self.textures_cache = textures
        
        # Check if file list changed
        current_files = set(textures.keys())
        if current_files != self.last_file_list:
            print(f"Loaded textures: {', '.join(sorted(textures.keys()))}")
            self.last_file_list = current_files
        
        # Create the tiled image (64x64 pixels)
        tiled = np.zeros((64, 64, 4), dtype=np.uint8)
        
        # Convert textures dict values to list for random selection
        texture_list = list(textures.values())
        
        # Fill each 8x8 tile with a randomly selected and rotated texture
        for row in range(8):
            for col in range(8):
                # Randomly select a texture
                selected_texture = random.choice(texture_list)
                
                # Random rotation (0, 90, 180, or 270 degrees)
                rotation = random.choice([0, 90, 180, 270])
                rotated = np.rot90(selected_texture, k=rotation//90)
                
                # Place the rotated tile
                y_start = row * 8
                x_start = col * 8
                tiled[y_start:y_start+8, x_start:x_start+8] = rotated
        
        return tiled
    
    def update_display(self):
        """Update the displayed image."""
        if not self.files_modified:
            return
        
        tiled_image = self.create_tiled_image()
        if tiled_image is not None:
            if self.im_display is None:
                self.im_display = self.ax.imshow(tiled_image)
            else:
                self.im_display.set_data(tiled_image)
            
            texture_count = len(self.textures_cache)
            self.ax.set_title(f"8x8 Tiled from {texture_count} texture{'s' if texture_count != 1 else ''}: {self.folder_path.name}/")
            self.fig.canvas.draw()
            self.files_modified = False
            print(f"Updated display at {time.strftime('%H:%M:%S')}")
    
    def mark_for_update(self):
        """Mark that files have been modified and need updating."""
        self.files_modified = True
    
    def randomize(self):
        """Force a new random arrangement of tiles."""
        self.files_modified = True
        self.update_display()

class FolderWatcher(FileSystemEventHandler):
    def __init__(self, tiler, watch_folder):
        self.tiler = tiler
        self.watch_folder = Path(watch_folder).resolve()
        
    def on_modified(self, event):
        # Check if it's a PNG file in our watched folder
        if not event.is_directory and event.src_path.endswith('.png'):
            event_path = Path(event.src_path)
            if event_path.parent.resolve() == self.watch_folder:
                print(f"Texture modified: {event_path.name}")
                self.tiler.mark_for_update()
    
    def on_created(self, event):
        # Handle new PNG files
        if not event.is_directory and event.src_path.endswith('.png'):
            event_path = Path(event.src_path)
            if event_path.parent.resolve() == self.watch_folder:
                print(f"New texture added: {event_path.name}")
                self.tiler.mark_for_update()
    
    def on_deleted(self, event):
        # Handle deleted PNG files
        if not event.is_directory and event.src_path.endswith('.png'):
            event_path = Path(event.src_path)
            if event_path.parent.resolve() == self.watch_folder:
                print(f"Texture removed: {event_path.name}")
                self.tiler.mark_for_update()

def main():
    if len(sys.argv) != 2:
        print("Usage: python multi_texture_tiler.py <path_to_texture_folder>")
        print("\nThe folder should contain 8x8 PNG images.")
        print("Press 'r' in the display window to randomize the tiles.")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    
    # Create the tiler
    try:
        tiler = MultiTextureTiler(folder_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    
    # Set up file watching
    event_handler = FolderWatcher(tiler, folder_path)
    observer = Observer()
    observer.schedule(event_handler, str(Path(folder_path).resolve()), recursive=False)
    observer.start()
    
    print(f"Watching '{folder_path}' for texture changes...")
    print("Press 'r' in the window to randomize tiles")
    print("Close the window to exit.")
    
    # Handle keyboard events
    def on_key(event):
        if event.key == 'r':
            print("Randomizing tiles...")
            tiler.randomize()
    
    tiler.fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Animation function to check for updates
    def animate(frame):
        tiler.update_display()
        return []
    
    # Set up animation to check for updates every 100ms
    anim = FuncAnimation(tiler.fig, animate, interval=100, blit=True)
    
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
        print("\nExiting...")

if __name__ == "__main__":
    main()