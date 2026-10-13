"""Standalone KP-RPE ViT-B face descriptor (CVLface, WebFace12M weights: non-commercial research use only).

    from kprpe_face import FaceRecognizer
    rec = FaceRecognizer()
    descr = rec(image_rgb, bboxes, landmarks)   # (N, 512) float32, L2-normalized
"""
from .recognizer import FaceRecognizer

__all__ = ['FaceRecognizer']
