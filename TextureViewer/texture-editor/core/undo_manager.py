"""Undo/Redo state management."""
from typing import List, Optional, Dict, Any
import copy
import numpy as np
from .pixel_data import PixelData


class CanvasState:
    """Represents a complete canvas state."""
    
    def __init__(self):
        self.variants: List[PixelData] = []
        self.variant_counts: List[int] = []
        self.current_variant_index: int = 0
        self.tile_assignments: Optional[np.ndarray] = None
        
    def capture_from(self, variant_manager):
        """Capture current state from variant manager."""
        # Deep copy all variants
        self.variants = [variant.copy() for variant in variant_manager.variants]
        self.variant_counts = variant_manager.tile_counts.copy()
        self.current_variant_index = variant_manager.current_variant_index
        if variant_manager.tile_assignments is not None:
            self.tile_assignments = variant_manager.tile_assignments.copy()
        else:
            self.tile_assignments = None
            
    def restore_to(self, variant_manager):
        """Restore this state to variant manager."""
        variant_manager.variants = [variant.copy() for variant in self.variants]
        variant_manager.tile_counts = self.variant_counts.copy()
        variant_manager.current_variant_index = self.current_variant_index
        if self.tile_assignments is not None:
            variant_manager.tile_assignments = self.tile_assignments.copy()
        else:
            variant_manager.tile_assignments = None


class UndoManager:
    """Manages undo/redo operations for the canvas."""
    
    def __init__(self, max_states: int = 250):
        self.max_states = max_states
        self.undo_stack: List[CanvasState] = []
        self.redo_stack: List[CanvasState] = []
        self.pending_state: Optional[CanvasState] = None
        self.has_changes = False
        
    def begin_operation(self, variant_manager):
        """Begin a new undoable operation."""
        if not self.has_changes:
            # Capture state at the beginning of operation
            self.pending_state = CanvasState()
            self.pending_state.capture_from(variant_manager)
            self.has_changes = False
            
    def mark_changed(self):
        """Mark that changes have been made in current operation."""
        self.has_changes = True
        
    def end_operation(self, variant_manager):
        """End the current operation and save state if changed."""
        if self.has_changes and self.pending_state is not None:
            # Save the pending state to undo stack
            self.undo_stack.append(self.pending_state)
            
            # Clear redo stack when new operation is recorded
            self.redo_stack.clear()
            
            # Limit undo stack size
            if len(self.undo_stack) > self.max_states:
                self.undo_stack.pop(0)
                
        self.pending_state = None
        self.has_changes = False
        
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self.undo_stack) > 0
        
    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(self.redo_stack) > 0
        
    def undo(self, variant_manager) -> bool:
        """Perform undo operation."""
        if not self.can_undo():
            return False
            
        # Save current state to redo stack
        current_state = CanvasState()
        current_state.capture_from(variant_manager)
        self.redo_stack.append(current_state)
        
        # Restore previous state
        previous_state = self.undo_stack.pop()
        previous_state.restore_to(variant_manager)
        
        return True
        
    def redo(self, variant_manager) -> bool:
        """Perform redo operation."""
        if not self.can_redo():
            return False
            
        # Save current state to undo stack
        current_state = CanvasState()
        current_state.capture_from(variant_manager)
        self.undo_stack.append(current_state)
        
        # Restore next state
        next_state = self.redo_stack.pop()
        next_state.restore_to(variant_manager)
        
        return True
        
    def clear(self):
        """Clear all undo/redo history."""
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.pending_state = None
        self.has_changes = False