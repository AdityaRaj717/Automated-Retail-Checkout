"""
Active Learning Module — Confidence-Gated Feedback Loop.

When the system is uncertain about a product (40-70% confidence), it:
1. Prompts the cashier via the UI to confirm or correct the prediction
2. Saves the confirmed/corrected image to the training dataset
3. Optionally triggers background fine-tuning

Research Value:
    - Handles dataset "drift" (e.g., Christmas edition Maggi)
    - Continuously improves recognition with real-world data
    - Implements a human-in-the-loop feedback cycle
"""

import os
import time
import threading
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class FeedbackItem:
    """An item pending cashier confirmation."""
    item_id: int
    suggested_name: str
    confidence: float
    cropped_image: Image.Image
    embedding: np.ndarray


class ActiveLearningManager:
    """
    Manages the active learning feedback loop between the AI and the cashier.

    Workflow:
        1. During inference, uncertain detections are flagged
        2. The UI displays a confirmation dialog to the cashier
        3. If confirmed, the image is saved to the dataset for future training
        4. Optionally, a background fine-tuning job is triggered

    Usage:
        manager = ActiveLearningManager(dataset_dir="dataset/processed")
        if manager.should_ask_cashier(confidence):
            manager.queue_feedback(item_id, name, conf, image, embedding)
        # Later, when cashier responds:
        manager.process_feedback(item_id, confirmed=True, corrected_name="maggi")
    """

    def __init__(self,
                 dataset_dir: str = "dataset/processed",
                 low_threshold: float = 0.40,
                 high_threshold: float = 0.70,
                 fine_tune_callback: Optional[Callable] = None):
        """
        Args:
            dataset_dir: Path to the processed dataset directory (ImageFolder structure)
            low_threshold: Minimum confidence to consider for active learning
            high_threshold: Maximum confidence — above this, auto-accept
            fine_tune_callback: Optional callable to trigger fine-tuning (runs in background)
        """
        self.dataset_dir = dataset_dir
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.fine_tune_callback = fine_tune_callback

        # Pending items awaiting cashier confirmation
        self._pending = {}  # {item_id: FeedbackItem}

        # Counter for triggering fine-tuning
        self._feedback_count = 0
        self._fine_tune_interval = 20  # Trigger fine-tune every N feedbacks

    def should_ask_cashier(self, confidence: float) -> bool:
        """
        Determine if the confidence level requires cashier confirmation.

        Args:
            confidence: Similarity score in [0, 1] range.

        Returns:
            True if confidence is in the ambiguous zone (40-70%).
        """
        return self.low_threshold <= confidence < self.high_threshold

    def queue_feedback(self, item_id: int, suggested_name: str,
                       confidence: float, cropped_image: Image.Image,
                       embedding: np.ndarray):
        """
        Queue an uncertain detection for cashier confirmation.

        Args:
            item_id: Unique identifier for this detection
            suggested_name: The AI's best guess
            confidence: Similarity score
            cropped_image: The cropped product image (PIL)
            embedding: The extracted embedding vector
        """
        self._pending[item_id] = FeedbackItem(
            item_id=item_id,
            suggested_name=suggested_name,
            confidence=confidence,
            cropped_image=cropped_image,
            embedding=embedding
        )
        print(f"[ACTIVE_LEARN] Queued item {item_id}: '{suggested_name}' "
              f"(confidence: {confidence:.1%}) for cashier review.")

    def process_feedback(self, item_id: int, confirmed: bool,
                         corrected_name: Optional[str] = None) -> bool:
        """
        Process cashier's response to a confirmation prompt.

        Args:
            item_id: The item being confirmed
            confirmed: True if cashier agreed with the suggestion
            corrected_name: If not confirmed, the corrected product name

        Returns:
            True if feedback was processed successfully
        """
        feedback = self._pending.pop(item_id, None)
        if feedback is None:
            print(f"[ACTIVE_LEARN] Warning: No pending item with id {item_id}")
            return False

        # Determine the final label
        final_label = feedback.suggested_name if confirmed else corrected_name
        if final_label is None:
            print("[ACTIVE_LEARN] No label provided for correction. Skipping.")
            return False

        # Save the image to the dataset
        saved = self._save_to_dataset(feedback.cropped_image, final_label)

        if saved:
            self._feedback_count += 1
            print(f"[ACTIVE_LEARN] Saved feedback #{self._feedback_count}: "
                  f"'{final_label}' (was: '{feedback.suggested_name}')")

            # Check if we should trigger fine-tuning
            if (self._feedback_count % self._fine_tune_interval == 0
                    and self.fine_tune_callback):
                self._trigger_fine_tune()

        return saved

    def _save_to_dataset(self, image: Image.Image, label: str) -> bool:
        """
        Save a confirmed image to the dataset directory structure.

        Saves as: dataset/processed/{label}/{timestamp}.png
        """
        try:
            label_dir = os.path.join(self.dataset_dir, label)
            os.makedirs(label_dir, exist_ok=True)

            # Generate unique filename
            timestamp = int(time.time() * 1000)
            filename = f"active_learn_{timestamp}.png"
            filepath = os.path.join(label_dir, filename)

            # Convert to RGBA if not already (to match training data format)
            if image.mode != 'RGBA':
                image = image.convert('RGBA')

            image.save(filepath)
            print(f"[ACTIVE_LEARN] Image saved to: {filepath}")
            return True

        except Exception as e:
            print(f"[ACTIVE_LEARN] Error saving image: {e}")
            return False

    def _trigger_fine_tune(self):
        """Trigger a background fine-tuning process."""
        if self.fine_tune_callback is None:
            return

        print("[ACTIVE_LEARN] Triggering background fine-tuning...")
        thread = threading.Thread(
            target=self.fine_tune_callback,
            daemon=True,
            name="active-learning-finetune"
        )
        thread.start()

    def get_pending_count(self) -> int:
        """Return number of items awaiting cashier confirmation."""
        return len(self._pending)

    def get_pending_item(self, item_id: int) -> Optional[FeedbackItem]:
        """Get a specific pending feedback item."""
        return self._pending.get(item_id)
