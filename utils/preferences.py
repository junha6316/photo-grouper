"""
User preferences management for the Photo Grouper application.
"""

import json
from pathlib import Path
from typing import Any, Dict


class Preferences:
    """Manage user preferences with persistent storage."""
    
    def __init__(self):
        """Initialize preferences with default values."""
        self.pref_dir = Path.home() / ".photo_grouper"
        self.pref_file = self.pref_dir / "preferences.json"
        self.defaults = {
            "export": {
                "open_folder_after": True,
                "organize_by_date": False,
                "copy_mode": True,  # True for copy, False for move
                "last_export_path": str(Path.home() / "Pictures")
            },
            "ui": {
                "thumbnail_size": 150,
                "default_view_mode": "grid"
            },
            "processing": {
                "similarity_threshold": 0.85,
                "min_group_size": 2
            }
        }
        self.prefs = self.load()
    
    def load(self) -> Dict:
        """Load preferences from disk."""
        if self.pref_file.exists():
            try:
                with open(self.pref_file, 'r') as f:
                    saved_prefs = json.load(f)
                    # Merge with defaults to handle new preferences
                    return self._merge_with_defaults(saved_prefs)
            except Exception as e:
                print(f"Error loading preferences: {e}")
                return self.defaults.copy()
        else:
            return self.defaults.copy()
    
    def _merge_with_defaults(self, saved: Dict) -> Dict:
        """Merge saved preferences with defaults."""
        merged = self.defaults.copy()
        
        def deep_update(base: Dict, update: Dict):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_update(base[key], value)
                else:
                    base[key] = value
        
        deep_update(merged, saved)
        return merged
    
    def save(self):
        """Save preferences to disk."""
        try:
            self.pref_dir.mkdir(exist_ok=True)
            with open(self.pref_file, 'w') as f:
                json.dump(self.prefs, f, indent=2)
        except Exception as e:
            print(f"Error saving preferences: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a preference value using dot notation (e.g., 'export.open_folder_after')."""
        keys = key.split('.')
        value = self.prefs
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set a preference value using dot notation."""
        keys = key.split('.')
        target = self.prefs
        
        # Navigate to the parent dict
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        # Set the value
        target[keys[-1]] = value
        self.save()
    
    def reset(self):
        """Reset all preferences to defaults."""
        self.prefs = self.defaults.copy()
        self.save()


# Global preferences instance
_preferences = None

def get_preferences() -> Preferences:
    """Get the global preferences instance."""
    global _preferences
    if _preferences is None:
        _preferences = Preferences()
    return _preferences