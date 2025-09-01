"""
Photo Select Session for deduplication of similar images.

Implements a pairwise comparison system where users choose between images
until one best image remains per group.
"""

import json
import os
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import copy

class ComparisonResult(Enum):
    """Result of a pairwise comparison."""
    LEFT_WINS = "left_wins"
    RIGHT_WINS = "right_wins"
    SKIP = "skip"

@dataclass
class ComparisonHistory:
    """Record of a single comparison decision."""
    group_index: int
    image_a: str
    image_b: str
    winner: str
    timestamp: float
    result: ComparisonResult

@dataclass
class GroupState:
    """State of a single group's comparison process."""
    original_images: List[str]
    remaining_images: List[str]
    current_index: int  # Index in remaining_images (current best image)
    compare_index: int  # Index in remaining_images (image being compared against)
    winner: Optional[str] = None
    skipped: bool = False
    completed: bool = False

@dataclass
class SessionState:
    """Complete state of a deduplication session."""
    session_id: str
    created_at: float
    last_updated: float
    current_group_index: int
    groups: List[GroupState]
    history: List[ComparisonHistory]
    similarity_threshold: float
    total_comparisons: int
    completed_comparisons: int

class SelectSession:
    """
    Manages a photo deduplication session using pairwise comparisons.
    
    For each group with similarity ≥ threshold:
    1. Start with first image as current best
    2. Compare current best with each remaining image
    3. User picks better image, which becomes new current best
    4. Continue until only one image remains
    """
    
    def __init__(self, groups: List[List[str]], similarity_threshold: float = 0.90, session_id: Optional[str] = None):
        """
        Initialize a new select session.
        
        Args:
            groups: List of image groups for deduplication
            similarity_threshold: Minimum similarity threshold for groups
            session_id: Optional existing session ID to resume
        """
        self.session_id = session_id or f"session_{int(time.time())}"
        self.similarity_threshold = similarity_threshold
        
        # Filter groups that need deduplication (more than 1 image)
        self.groups = [group for group in groups if len(group) > 1]
        
        # Initialize session state
        self.state = SessionState(
            session_id=self.session_id,
            created_at=time.time(),
            last_updated=time.time(),
            current_group_index=0,
            groups=[self._create_group_state(group) for group in self.groups],
            history=[],
            similarity_threshold=similarity_threshold,
            total_comparisons=self._calculate_total_comparisons(),
            completed_comparisons=0
        )
    
    def _create_group_state(self, images: List[str]) -> GroupState:
        """Create initial state for a group."""
        return GroupState(
            original_images=images.copy(),
            remaining_images=images.copy(),
            current_index=0,
            compare_index=1 if len(images) > 1 else 0,
            winner=None,
            skipped=False,
            completed=len(images) <= 1
        )
    
    def _calculate_total_comparisons(self) -> int:
        """Calculate total number of comparisons needed."""
        total = 0
        for group in self.groups:
            if len(group) > 1:
                total += len(group) - 1  # n-1 comparisons for n images
        return total
    
    def get_current_comparison(self) -> Optional[Tuple[str, str, int, int]]:
        """
        Get the current image comparison.
        
        Returns:
            Tuple of (image_a_path, image_b_path, group_index, total_groups)
            or None if session is complete
        """
        if self.is_session_complete():
            return None
        
        # Find next group that needs comparison
        while self.state.current_group_index < len(self.state.groups):
            group = self.state.groups[self.state.current_group_index]
            
            if group.completed or group.skipped:
                self.state.current_group_index += 1
                continue
            
            if len(group.remaining_images) <= 1:
                # Mark group as completed
                group.completed = True
                if group.remaining_images:
                    group.winner = group.remaining_images[0]
                self.state.current_group_index += 1
                continue
            
            # Safety check - ensure indices are valid
            if (group.current_index >= len(group.remaining_images) or 
                group.compare_index >= len(group.remaining_images)):
                # Reset indices if they're invalid
                group.current_index = 0
                group.compare_index = 1 if len(group.remaining_images) > 1 else 0
            
            # Double check that indices are different and valid
            if group.current_index == group.compare_index and len(group.remaining_images) > 1:
                group.compare_index = (group.current_index + 1) % len(group.remaining_images)
            
            # Get current images for comparison
            image_a = group.remaining_images[group.current_index]
            image_b = group.remaining_images[group.compare_index]
            
            return image_a, image_b, self.state.current_group_index, len(self.state.groups)
        
        return None
    
    def make_comparison(self, result: ComparisonResult) -> bool:
        """
        Process a comparison result and advance the session.
        
        Args:
            result: The comparison result
            
        Returns:
            True if comparison was processed, False if session is complete
        """
        if self.is_session_complete():
            return False
        
        group = self.state.groups[self.state.current_group_index]
        
        if group.completed or group.skipped:
            return False
        
        # Safety check - ensure we have enough images and valid indices
        if len(group.remaining_images) <= 1:
            group.completed = True
            if group.remaining_images:
                group.winner = group.remaining_images[0]
            return False
        
        # Ensure indices are valid
        if (group.current_index >= len(group.remaining_images) or 
            group.compare_index >= len(group.remaining_images)):
            return False
        
        image_a = group.remaining_images[group.current_index]
        image_b = group.remaining_images[group.compare_index]
        
        # Record the comparison in history
        history_entry = ComparisonHistory(
            group_index=self.state.current_group_index,
            image_a=image_a,
            image_b=image_b,
            winner="",
            timestamp=time.time(),
            result=result
        )
        
        if result == ComparisonResult.SKIP:
            group.skipped = True
            history_entry.winner = "skipped"
        elif result == ComparisonResult.LEFT_WINS:
            # Left image (current) wins, remove right image
            winner = image_a
            group.remaining_images.pop(group.compare_index)
            history_entry.winner = winner
            self.state.completed_comparisons += 1
            
            # Adjust current index if compare was before it
            if group.compare_index < group.current_index:
                group.current_index -= 1
        elif result == ComparisonResult.RIGHT_WINS:
            # Right image wins, remove left image and right becomes new current
            winner = image_b
            compare_idx = group.compare_index
            group.remaining_images.pop(group.current_index)
            
            # Adjust compare index if current was before it
            if group.current_index < compare_idx:
                compare_idx -= 1
            
            group.current_index = compare_idx
            history_entry.winner = winner
            self.state.completed_comparisons += 1
        
        self.state.history.append(history_entry)
        
        # Update compare index for next comparison
        if not group.skipped and len(group.remaining_images) > 1:
            # Find next image to compare (next image that isn't the current best)
            group.compare_index = 0
            if group.compare_index == group.current_index:
                group.compare_index = 1
            
            # Safety check - ensure indices are valid
            if group.current_index >= len(group.remaining_images):
                group.current_index = 0
            if group.compare_index >= len(group.remaining_images):
                group.compare_index = (group.current_index + 1) % len(group.remaining_images) if len(group.remaining_images) > 1 else 0
        
        # Check if group is complete
        if len(group.remaining_images) <= 1:
            group.completed = True
            if group.remaining_images:
                group.winner = group.remaining_images[0]
        
        self.state.last_updated = time.time()
        return True
    
    def undo_last_comparison(self) -> bool:
        """
        Undo the last comparison decision.
        
        Returns:
            True if undo was successful, False if no history available
        """
        if not self.state.history:
            return False
        
        last_entry = self.state.history.pop()
        group = self.state.groups[last_entry.group_index]
        
        # Reset group to before the last comparison
        if last_entry.result == ComparisonResult.SKIP:
            group.skipped = False
            # Reset to the group this comparison was from
            self.state.current_group_index = last_entry.group_index
        else:
            # Restore the removed image
            if last_entry.result == ComparisonResult.LEFT_WINS:
                # Right image was removed, restore it
                group.remaining_images.append(last_entry.image_b)
            elif last_entry.result == ComparisonResult.RIGHT_WINS:
                # Left image was removed, restore it  
                group.remaining_images.append(last_entry.image_a)
            
            # Reset group state to before comparison
            group.completed = False
            group.winner = None
            self.state.completed_comparisons -= 1
            
            # Reset to the group this comparison was from
            self.state.current_group_index = last_entry.group_index
            
            # Restore the exact image indices
            try:
                group.current_index = group.remaining_images.index(last_entry.image_a)
                group.compare_index = group.remaining_images.index(last_entry.image_b)
            except ValueError:
                # Fallback: reset to start of group
                group.current_index = 0
                group.compare_index = 1 if len(group.remaining_images) > 1 else 0
        
        self.state.last_updated = time.time()
        return True
    
    def skip_current_group(self) -> bool:
        """
        Skip the current group and move to the next one.
        
        Returns:
            True if group was skipped, False if no current group
        """
        comparison = self.get_current_comparison()
        if comparison is None:
            return False
        
        return self.make_comparison(ComparisonResult.SKIP)
    
    def is_session_complete(self) -> bool:
        """Check if the session is complete."""
        return self.state.current_group_index >= len(self.state.groups)
    
    def get_progress(self) -> Tuple[int, int, float]:
        """
        Get session progress.
        
        Returns:
            Tuple of (completed_comparisons, total_comparisons, percentage)
        """
        completed = self.state.completed_comparisons
        total = self.state.total_comparisons
        percentage = (completed / total * 100) if total > 0 else 100
        return completed, total, percentage
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get session results.
        
        Returns:
            Dictionary containing session results and statistics
        """
        completed_groups = [g for g in self.state.groups if g.completed and not g.skipped]
        skipped_groups = [g for g in self.state.groups if g.skipped]
        
        winners = []
        for group in completed_groups:
            if group.winner:
                winners.append({
                    'winner': group.winner,
                    'original_group': group.original_images,
                    'eliminated': [img for img in group.original_images if img != group.winner]
                })
        
        return {
            'session_id': self.state.session_id,
            'total_groups': len(self.state.groups),
            'completed_groups': len(completed_groups),
            'skipped_groups': len(skipped_groups),
            'winners': winners,
            'total_comparisons': self.state.total_comparisons,
            'completed_comparisons': self.state.completed_comparisons,
            'similarity_threshold': self.state.similarity_threshold,
            'duration': self.state.last_updated - self.state.created_at,
            'history': self.state.history
        }
    
    def save_session(self, filepath: str) -> bool:
        """
        Save session state to file.
        
        Args:
            filepath: Path to save the session file
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Convert dataclasses to dict for JSON serialization
            session_dict = asdict(self.state)
            
            # Convert enum values to strings
            for entry in session_dict['history']:
                entry['result'] = entry['result'].value
            
            with open(filepath, 'w') as f:
                json.dump(session_dict, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving session: {e}")
            return False
    
    @classmethod
    def load_session(cls, filepath: str) -> Optional['SelectSession']:
        """
        Load session state from file.
        
        Args:
            filepath: Path to the session file
            
        Returns:
            SelectSession instance or None if loading failed
        """
        try:
            with open(filepath, 'r') as f:
                session_dict = json.load(f)
            
            # Convert string enum values back to enums
            for entry in session_dict['history']:
                entry['result'] = ComparisonResult(entry['result'])
            
            # Create session instance
            session = cls([], session_dict['similarity_threshold'], session_dict['session_id'])
            
            # Restore state
            session.state = SessionState(**session_dict)
            
            # Convert history back to dataclass objects
            session.state.history = [
                ComparisonHistory(**entry) for entry in session.state.history
            ]
            
            # Convert groups back to dataclass objects
            session.state.groups = [
                GroupState(**group) for group in session.state.groups
            ]
            
            return session
        except Exception as e:
            print(f"Error loading session: {e}")
            return None
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get basic session information."""
        return {
            'session_id': self.state.session_id,
            'created_at': self.state.created_at,
            'last_updated': self.state.last_updated,
            'total_groups': len(self.state.groups),
            'similarity_threshold': self.state.similarity_threshold,
            'progress': self.get_progress(),
            'is_complete': self.is_session_complete()
        }