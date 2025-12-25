"""
Selection review session state for swipe mode.
"""

from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import hashlib
import json
import time


class ReviewDecision(str, Enum):
    KEEP = "keep"
    REJECT = "reject"


class ReviewSession:
    def __init__(
        self,
        folder_path: str,
        image_paths: List[str],
        session_id: Optional[str] = None,
    ):
        self.folder_path = folder_path or ""
        self.image_paths = list(image_paths)
        self._index_by_path = {path: idx for idx, path in enumerate(self.image_paths)}
        self.session_id = session_id or self._make_session_id(
            self.folder_path, self.image_paths
        )
        self.storage_path = self._build_storage_path(self.session_id)

        self.decisions: Dict[str, str] = {}
        self.history: List[Tuple[int, str, Optional[str], str]] = []
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.current_index = 0
        self._saved_paths: List[str] = []
        self._saved_index = 0

        self._load_state()
        self._prune_decisions()
        self._restore_index()

    @staticmethod
    def _make_session_id(folder_path: str, image_paths: Iterable[str]) -> str:
        digest = hashlib.md5()
        seed = folder_path or "default"
        digest.update(seed.encode("utf-8"))
        digest.update(b"\0")
        for path in sorted(image_paths):
            digest.update(path.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    @staticmethod
    def _build_storage_path(session_id: str) -> Path:
        base_dir = Path.home() / ".photo_grouper" / "review_sessions"
        return base_dir / f"{session_id}.json"

    def _load_state(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            with open(self.storage_path, "r") as handle:
                data = json.load(handle)
        except Exception as exc:
            print(f"Error loading review session: {exc}")
            return

        self.created_at = data.get("created_at", self.created_at)
        self.updated_at = data.get("updated_at", self.updated_at)
        self.decisions = data.get("decisions", {}) or {}
        self._saved_paths = data.get("image_paths", []) or []
        self._saved_index = int(data.get("current_index", 0))

    def _prune_decisions(self) -> None:
        if not self.decisions:
            return
        image_set = set(self.image_paths)
        self.decisions = {
            path: decision
            for path, decision in self.decisions.items()
            if path in image_set
        }

    def _restore_index(self) -> None:
        start_index = 0
        if self._saved_paths and 0 <= self._saved_index < len(self._saved_paths):
            saved_path = self._saved_paths[self._saved_index]
            start_index = self._index_by_path.get(saved_path, 0)
        self.current_index = start_index
        next_index = self._find_next_undecided(start_index)
        if next_index is None:
            self.current_index = len(self.image_paths)
        else:
            self.current_index = next_index

    def _find_next_undecided(self, start_index: int) -> Optional[int]:
        if not self.image_paths:
            return None
        start_index = max(0, min(start_index, len(self.image_paths)))
        for idx in range(start_index, len(self.image_paths)):
            if self.image_paths[idx] not in self.decisions:
                return idx
        for idx in range(0, start_index):
            if self.image_paths[idx] not in self.decisions:
                return idx
        return None

    def get_current_image(self) -> Optional[str]:
        if 0 <= self.current_index < len(self.image_paths):
            return self.image_paths[self.current_index]
        return None

    def get_decision(self, image_path: str) -> Optional[str]:
        return self.decisions.get(image_path)

    def seed_decisions(
        self,
        image_paths: Iterable[str],
        decision: ReviewDecision = ReviewDecision.KEEP,
    ) -> bool:
        decision_value = decision.value if isinstance(decision, ReviewDecision) else str(decision)
        changed = False
        for path in image_paths:
            if path in self._index_by_path and path not in self.decisions:
                self.decisions[path] = decision_value
                changed = True
        if changed:
            self.updated_at = time.time()
        return changed

    def set_decision(self, image_path: str, decision: ReviewDecision) -> bool:
        if image_path not in self._index_by_path:
            return False
        decision_value = decision.value if isinstance(decision, ReviewDecision) else str(decision)
        previous = self.decisions.get(image_path)
        if previous == decision_value:
            return False
        self.decisions[image_path] = decision_value
        index = self._index_by_path[image_path]
        self.history.append((index, image_path, previous, decision_value))
        self.updated_at = time.time()
        return True

    def advance(self) -> bool:
        next_index = self._find_next_undecided(self.current_index + 1)
        if next_index is None:
            self.current_index = len(self.image_paths)
            return False
        self.current_index = next_index
        return True

    def undo_last(self) -> Optional[Tuple[str, Optional[str]]]:
        if not self.history:
            return None
        index, image_path, previous, _new_value = self.history.pop()
        if previous is None:
            self.decisions.pop(image_path, None)
        else:
            self.decisions[image_path] = previous
        self.current_index = index
        self.updated_at = time.time()
        return image_path, previous

    def get_progress(self) -> Tuple[int, int, int, int]:
        total = len(self.image_paths)
        decided = len(self.decisions)
        kept = sum(
            1 for value in self.decisions.values()
            if value == ReviewDecision.KEEP.value
        )
        rejected = decided - kept
        return decided, total, kept, rejected

    def save(self) -> bool:
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "session_id": self.session_id,
                "folder_path": self.folder_path,
                "created_at": self.created_at,
                "updated_at": time.time(),
                "image_paths": self.image_paths,
                "decisions": self.decisions,
                "current_index": self.current_index,
            }
            with open(self.storage_path, "w") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=True)
            self.updated_at = payload["updated_at"]
            return True
        except Exception as exc:
            print(f"Error saving review session: {exc}")
            return False
