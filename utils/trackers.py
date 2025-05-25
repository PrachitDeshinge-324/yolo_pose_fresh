import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
import numpy as np

@dataclass
class Detection:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    track_id: Optional[int] = None
    
class YOLOTracker:
    """Wrapper for YOLO's built-in tracking"""
    def __init__(self, model, device='cpu', min_confidence=0.45, classes=[0]):
        self.model = model
        self.device = device
        self.min_confidence = min_confidence
        self.classes = classes
        self.id_mapping = {}
        self.next_track_id = 1
        
    def update(self, frame):
        """Track objects in frame using YOLO's built-in tracker"""
        results = self.model.track(frame, persist=True, conf=self.min_confidence, 
                                  classes=self.classes, device=self.device, verbose=False)
        
        detections = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            track_ids = results[0].boxes.id
            
            if track_ids is not None:
                track_ids = track_ids.int().cpu().numpy()
                for i, (box, conf, yolo_id) in enumerate(zip(boxes, confs, track_ids)):
                    if yolo_id not in self.id_mapping:
                        self.id_mapping[yolo_id] = self.next_track_id
                        self.next_track_id += 1
                    track_id = self.id_mapping[yolo_id]
                    
                    detections.append(Detection(
                        bbox=box,
                        confidence=conf,
                        track_id=track_id
                    ))
        
        return detections
