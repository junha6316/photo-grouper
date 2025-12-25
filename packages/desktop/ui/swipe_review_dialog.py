"""
Swipe-based selection review dialog.
"""

import os
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QKeySequence, QShortcut, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from core.review_session import ReviewDecision, ReviewSession
from ui.utils.image_utils import load_image_as_pixmap

NEAR_DUPLICATE_THRESHOLD = 0.98


class SwipeImageView(QFrame):
    swiped_left = Signal()
    swiped_right = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._press_pos = QPoint()
        self._dragging = False
        self._drag_offset = QPoint()
        self._swipe_threshold = 120

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.image_label.setStyleSheet("background-color: #fdfdfd;")

        self.setStyleSheet("""
            QFrame {
                background-color: #fafafa;
                border: 1px solid #ddd;
                border-radius: 10px;
            }
        """)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.image_label.resize(self.size())
        self._update_pixmap()
        self._apply_offset()

    def set_image(self, image_path: str) -> None:
        self._pixmap = None
        self.image_label.setText("Loading image...")
        self._pixmap = load_image_as_pixmap(image_path, max_size=1400)
        if self._pixmap is None:
            self.image_label.setText("Error loading image.")
        self._drag_offset = QPoint()
        self._apply_offset()
        self._update_pixmap()

    def show_message(self, message: str) -> None:
        self._pixmap = None
        self.image_label.setText(message)
        self._drag_offset = QPoint()
        self._apply_offset()
        self._update_pixmap()

    def _update_pixmap(self) -> None:
        if self._pixmap is None:
            self.image_label.setPixmap(QPixmap())
            return
        target_width = max(1, self.width() - 40)
        target_height = max(1, self.height() - 40)
        scaled = self._pixmap.scaled(
            target_width,
            target_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.image_label.setText("")

    def _apply_offset(self) -> None:
        self.image_label.move(self._drag_offset)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._press_pos = event.position().toPoint()
            self._dragging = True
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            delta = event.position().toPoint() - self._press_pos
            self._drag_offset = delta
            self._apply_offset()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if not self._dragging:
            super().mouseReleaseEvent(event)
            return
        delta = event.position().toPoint() - self._press_pos
        self._dragging = False
        self._drag_offset = QPoint()
        self._apply_offset()

        if abs(delta.x()) > self._swipe_threshold and abs(delta.x()) > abs(delta.y()):
            if delta.x() < 0:
                self.swiped_left.emit()
            else:
                self.swiped_right.emit()
        super().mouseReleaseEvent(event)


class SwipeReviewDialog(QDialog):
    def __init__(self, session: ReviewSession, parent=None, title: Optional[str] = None):
        super().__init__(parent)
        self.session = session
        self.main_window = parent
        self.title_text = title or "Selection Mode"
        self.similarity_threshold = NEAR_DUPLICATE_THRESHOLD
        self._selected_set_cache = set()
        self._selected_embeddings = {}
        self._embeddings_unavailable = False

        self.setWindowTitle(self.title_text)
        self.setMinimumSize(1000, 700)

        self.init_ui()
        self.setup_shortcuts()
        self._seed_from_selection()
        self.load_current_image()

    def init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header_layout = QHBoxLayout()
        self.title_label = QLabel(self.title_text)
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
        header_layout.addWidget(self.title_label)

        header_layout.addStretch()

        self.progress_label = QLabel()
        self.progress_label.setStyleSheet("font-size: 12px; color: #666;")
        header_layout.addWidget(self.progress_label)

        self.counts_label = QLabel()
        self.counts_label.setStyleSheet("font-size: 12px; color: #666; margin-left: 12px;")
        header_layout.addWidget(self.counts_label)

        layout.addLayout(header_layout)

        self.image_view = SwipeImageView()
        self.image_view.swiped_left.connect(self.reject_current)
        self.image_view.swiped_right.connect(self.keep_current)
        layout.addWidget(self.image_view, 1)

        self.filename_label = QLabel("")
        self.filename_label.setStyleSheet("font-size: 12px; color: #555;")
        self.filename_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.filename_label)

        self.similarity_label = QLabel("")
        self.similarity_label.setStyleSheet("font-size: 12px; color: #555;")
        self.similarity_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.similarity_label)

        self.help_label = QLabel("Drag left/right or use arrow keys. U or Backspace to undo.")
        self.help_label.setStyleSheet("font-size: 11px; color: #777;")
        self.help_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.help_label)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)

        self.reject_button = QPushButton("Reject (Left)")
        self.reject_button.clicked.connect(self.reject_current)
        self.reject_button.setStyleSheet("""
            QPushButton {
                padding: 10px 18px;
                font-size: 12px;
                border: 2px solid #d32f2f;
                border-radius: 6px;
                background-color: #fdecea;
                color: #b71c1c;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #f9d6d1;
            }
        """)
        button_layout.addWidget(self.reject_button)

        self.undo_button = QPushButton("Undo (U)")
        self.undo_button.clicked.connect(self.undo_last)
        self.undo_button.setStyleSheet("""
            QPushButton {
                padding: 10px 16px;
                font-size: 12px;
                border: 1px solid #ffa000;
                border-radius: 6px;
                background-color: #fff8e1;
                color: #e65100;
            }
            QPushButton:hover {
                background-color: #ffecb3;
            }
        """)
        button_layout.addWidget(self.undo_button)

        self.keep_button = QPushButton("Keep (Right)")
        self.keep_button.clicked.connect(self.keep_current)
        self.keep_button.setStyleSheet("""
            QPushButton {
                padding: 10px 18px;
                font-size: 12px;
                border: 2px solid #2e7d32;
                border-radius: 6px;
                background-color: #e8f5e9;
                color: #1b5e20;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c8e6c9;
            }
        """)
        button_layout.addWidget(self.keep_button)

        button_layout.addStretch()

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        self.close_button.setStyleSheet("""
            QPushButton {
                padding: 10px 16px;
                font-size: 12px;
                border: 1px solid #ccc;
                border-radius: 6px;
                background-color: #f5f5f5;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)

    def setup_shortcuts(self) -> None:
        left_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self)
        left_shortcut.activated.connect(self.reject_current)

        right_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self)
        right_shortcut.activated.connect(self.keep_current)

        undo_shortcut = QShortcut(QKeySequence(Qt.Key_U), self)
        undo_shortcut.activated.connect(self.undo_last)

        backspace_shortcut = QShortcut(QKeySequence(Qt.Key_Backspace), self)
        backspace_shortcut.activated.connect(self.undo_last)

        escape_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        escape_shortcut.activated.connect(self.close)

    def _seed_from_selection(self) -> None:
        if not self.main_window or not hasattr(self.main_window, "global_selected_images"):
            return
        changed = self.session.seed_decisions(self.main_window.global_selected_images)
        if changed:
            self.session.save()
        current_image = self.session.get_current_image()
        if current_image and self.session.get_decision(current_image):
            self.session.advance()
        self._sync_selection_from_decisions()

    def _sync_selection_from_decisions(self) -> None:
        if not self.main_window or not hasattr(self.main_window, "global_selected_images"):
            return
        selected_now = self.main_window.get_global_selected_images()
        keep_set = {
            path for path, decision in self.session.decisions.items()
            if decision == ReviewDecision.KEEP.value
        }
        reject_set = {
            path for path, decision in self.session.decisions.items()
            if decision == ReviewDecision.REJECT.value
        }

        for image_path in sorted(keep_set - selected_now):
            self.main_window.on_selection_changed_from_view(image_path, True)

        for image_path in sorted(reject_set & selected_now):
            self.main_window.on_selection_changed_from_view(image_path, False)

    def load_current_image(self) -> None:
        image_path = self.session.get_current_image()
        if not image_path:
            self.image_view.show_message("All images reviewed.")
            self.filename_label.setText("")
            self.similarity_label.setText("")
            self._update_progress()
            self._set_action_buttons_enabled(False)
            return
        self.image_view.set_image(image_path)
        self.filename_label.setText(os.path.basename(image_path))
        self._update_similarity_indicator(image_path)
        self._update_progress()
        self._set_action_buttons_enabled(True)

    def _update_progress(self) -> None:
        decided, total, kept, rejected = self.session.get_progress()
        if total == 0:
            self.progress_label.setText("No images to review.")
            self.counts_label.setText("")
            return
        self.progress_label.setText(f"Decided {decided}/{total}")
        self.counts_label.setText(f"Keep {kept} | Reject {rejected}")

    def _set_action_buttons_enabled(self, enabled: bool) -> None:
        self.keep_button.setEnabled(enabled)
        self.reject_button.setEnabled(enabled)
        self.undo_button.setEnabled(bool(self.session.history))

    def _get_selected_images(self) -> set:
        if not self.main_window or not hasattr(self.main_window, "get_global_selected_images"):
            return set()
        return set(self.main_window.get_global_selected_images())

    def _sync_selected_embeddings(self, selected_set: set) -> None:
        if self._embeddings_unavailable:
            return
        if not self.main_window or not hasattr(self.main_window, "current_embeddings"):
            self._embeddings_unavailable = True
            return
        if selected_set == self._selected_set_cache:
            return
        embeddings = self.main_window.current_embeddings
        removed = self._selected_set_cache - selected_set
        added = selected_set - self._selected_set_cache
        for path in removed:
            self._selected_embeddings.pop(path, None)
        for path in added:
            embedding = embeddings.get(path)
            if embedding is not None:
                self._selected_embeddings[path] = embedding
        self._selected_set_cache = selected_set

    def _max_similarity_to_selected(self, image_path: str, selected_set: set) -> Optional[float]:
        if not selected_set:
            return None
        if not self.main_window or not hasattr(self.main_window, "current_embeddings"):
            return None
        target = self.main_window.current_embeddings.get(image_path)
        if target is None:
            return None
        self._sync_selected_embeddings(selected_set)
        if self._embeddings_unavailable:
            return None
        max_sim = None
        for path in selected_set:
            if path == image_path:
                continue
            embedding = self._selected_embeddings.get(path)
            if embedding is None:
                continue
            similarity = float(np.dot(target, embedding))
            if max_sim is None or similarity > max_sim:
                max_sim = similarity
        return max_sim

    def _update_similarity_indicator(self, image_path: str) -> None:
        selected_set = self._get_selected_images()
        selected_others = selected_set - {image_path}
        if not selected_others:
            self.similarity_label.setStyleSheet("font-size: 12px; color: #777;")
            self.similarity_label.setText(
                f"Selected photos >= {self.similarity_threshold:.2f}: none"
            )
            return
        max_sim = self._max_similarity_to_selected(image_path, selected_set)
        if max_sim is None:
            self.similarity_label.setStyleSheet("font-size: 12px; color: #777;")
            self.similarity_label.setText(
                f"Selected photos >= {self.similarity_threshold:.2f}: unknown"
            )
            return
        if max_sim >= self.similarity_threshold:
            color = "#2e7d32"
            status = "yes"
        else:
            color = "#c62828"
            status = "no"
        self.similarity_label.setStyleSheet(f"font-size: 12px; color: {color};")
        self.similarity_label.setText(
            f"Selected photos >= {self.similarity_threshold:.2f}: {status} (max {max_sim:.3f})"
        )

    def _apply_decision(self, decision: ReviewDecision) -> None:
        image_path = self.session.get_current_image()
        if not image_path:
            return
        changed = self.session.set_decision(image_path, decision)
        if changed:
            self._apply_selection_change(image_path, decision)
            self.session.save()
        self.session.advance()
        self.load_current_image()

    def _apply_selection_change(self, image_path: str, decision: ReviewDecision) -> None:
        if not self.main_window:
            return
        is_selected = decision == ReviewDecision.KEEP
        self.main_window.on_selection_changed_from_view(image_path, is_selected)

    def keep_current(self) -> None:
        self._apply_decision(ReviewDecision.KEEP)

    def reject_current(self) -> None:
        self._apply_decision(ReviewDecision.REJECT)

    def undo_last(self) -> None:
        undo_result = self.session.undo_last()
        if not undo_result:
            self._set_action_buttons_enabled(self.session.get_current_image() is not None)
            return
        image_path, restored = undo_result
        if self.main_window:
            is_selected = restored == ReviewDecision.KEEP.value
            self.main_window.on_selection_changed_from_view(image_path, is_selected)
        self.session.save()
        self.load_current_image()

    def closeEvent(self, event):
        self.session.save()
        super().closeEvent(event)
