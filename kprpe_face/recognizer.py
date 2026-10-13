"""FaceRecognizer: image + boxes + 5 landmarks of every face -> KP-RPE ViT-B face descriptors.

Pipeline per face: similarity transform of the 5 landmarks onto the ArcFace 112x112 template (least squares,
Umeyama), cv2.warpAffine with a black border; the same 5 landmarks mapped into the crop and scaled to [0, 1] go to
the network as the second input. All faces of one image run through the network as one batch.
"""
import os
import warnings

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

from .download import SHA256 as WEIGHTS_SHA256, resolve as resolve_weights, sha256_file
from .model import ViTKPRPE

CKPT_PREFIX = 'model.net.'

CROP = 112
# ArcFace 5-point template of the 112x112 crop (InsightFace, MIT): left eye, right eye, nose, left and right mouth
# corner, "left" = image left
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], np.float64)
FLIP_ORDER = [1, 0, 2, 4, 3]  # landmark order after a horizontal flip
DESCR_DIM = 512


def umeyama(src, dst):
    """Least-squares similarity transform src -> dst. Returns 2x3.

    Port of skimage.transform._geometric._umeyama (scikit-image, BSD-3-Clause, see THIRD_PARTY_NOTICES.md)."""
    num, dim = src.shape
    src_mean, dst_mean = src.mean(0), dst.mean(0)
    src_d, dst_d = src - src_mean, dst - dst_mean
    A = dst_d.T @ src_d / num
    d = np.ones(dim)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
    T = np.eye(dim + 1)
    U, S, V = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.full((2, 3), np.nan)
    if rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V
    scale = 1.0 / src_d.var(axis=0).sum() * (S @ d)
    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale
    return T[:2]


def _pick_device(device):
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class FaceRecognizer:
    """KP-RPE ViT-B (CVLface, WebFace12M) face descriptor. Weights: non-commercial research use only.

        rec = FaceRecognizer()                              # device: cuda > mps > cpu
        descr = rec(image, bboxes, landmarks)               # (N, 512) float32, L2-normalized
        descr, norms = rec(image, bboxes, landmarks, return_norms=True)

    weights    path to model.safetensors, or None: next to the code, else the cache, downloaded there on first use
               (see kprpe_face.download; download=False raises instead of downloading).

    image      HxWx3 uint8, RGB (or BGR with input_color='bgr').
    bboxes     (N, 4+) x1, y1, x2, y2[, score, ...] in pixels, or None. Not used by the model (alignment needs only
               the landmarks); checked only for count and as a sanity check that the landmarks lie near their box.
    landmarks  (N, 5, 2) (x, y) in pixels of this image, order: left eye, right eye, nose, left mouth corner,
               right mouth corner ("left" = image left); or (N, 10) interleaved x0, y0, x1, y1, ... (detector format).
    norms      L2 norm of the raw embedding (higher = more confident); a norm-weighted mean of the descriptors
               (= mean of the raw embeddings) is a simple way to aggregate a track.

    All faces of one image are embedded as one batch (split into chunks of batch_size only if there are more faces).
    flip=True adds horizontal-flip test-time augmentation (sum of the embeddings of the crop and its mirror).
    """

    def __init__(self, weights=None, device=None, batch_size=64, flip=False, input_color='rgb',
                 check_sha256=True, download=True):
        if input_color not in ('rgb', 'bgr'):
            raise ValueError("input_color must be 'rgb' or 'bgr'")
        if weights is None:
            weights = resolve_weights(download_missing=download)
        elif not os.path.isfile(weights):
            raise FileNotFoundError(f'weights not found: {weights}')
        self.weights = weights
        if check_sha256 and sha256_file(weights) != WEIGHTS_SHA256:
            raise ValueError(f'unexpected weights file (sha256 mismatch): {weights}')
        state = load_file(weights)
        state = {k[len(CKPT_PREFIX):]: v for k, v in state.items() if k.startswith(CKPT_PREFIX)}
        model = ViTKPRPE()
        model.load_state_dict(state, strict=True)
        self.device = _pick_device(device)
        self.model = model.eval().to(self.device)
        self.batch_size, self.flip, self.input_color = int(batch_size), bool(flip), input_color

    # ------------------------------------------------------------------------------------------------------------
    def __call__(self, image, bboxes, landmarks, return_norms=False):
        pts = self._landmarks(landmarks)
        if bboxes is not None:
            self._check_boxes(bboxes, pts)
        if len(pts) == 0:
            empty = np.zeros((0, DESCR_DIM), np.float32)
            return (empty, np.zeros(0)) if return_norms else empty
        crops, keypoints = self.align(image, pts)
        raw = self.embed_aligned(crops, keypoints).astype(np.float64)
        norms = np.linalg.norm(raw, axis=1)
        descr = (raw / norms[:, None]).astype(np.float32)
        return (descr, norms) if return_norms else descr

    def align(self, image, landmarks):
        """-> crops (N, 112, 112, 3) uint8 RGB, keypoints (N, 5, 2) float32: landmarks in the crop, (x, y) in [0, 1]."""
        image = self._image(image)
        pts = self._landmarks(landmarks)
        crops = np.empty((len(pts), CROP, CROP, 3), np.uint8)
        keypoints = np.empty((len(pts), 5, 2), np.float32)
        for i, p in enumerate(pts):
            M = umeyama(p, ARCFACE_DST * CROP / 112.)
            if not np.isfinite(M).all():
                raise ValueError(f'face {i}: degenerate landmarks {p.tolist()}')
            crops[i] = cv2.warpAffine(image, M, (CROP, CROP), borderValue=0.0)
            keypoints[i] = (p @ M[:, :2].T + M[:, 2]) / CROP
        if self.input_color == 'bgr':
            crops = np.ascontiguousarray(crops[..., ::-1])
        return crops, keypoints

    @torch.no_grad()
    def embed_aligned(self, crops, keypoints):
        """Aligned RGB crops (N, 112, 112, 3) uint8 + keypoints (N, 5, 2) -> raw embeddings (N, 512) float32."""
        out = []
        for i in range(0, len(crops), self.batch_size):
            x = torch.from_numpy(crops[i:i + self.batch_size]).float().permute(0, 3, 1, 2).contiguous()
            x = ((x / 255. - 0.5) / 0.5).to(self.device)
            k = torch.from_numpy(np.ascontiguousarray(keypoints[i:i + self.batch_size], np.float32)).to(self.device)
            e = self.model(x, k)
            if self.flip:
                kf = k.clone()
                kf[..., 0] = 1 - kf[..., 0]
                kf = kf[:, FLIP_ORDER]
                e = e + self.model(x.flip(3), kf)
            out.append(e.float().cpu().numpy())
        return np.concatenate(out)

    # ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _image(image):
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f'image must be HxWx3 uint8, got {image.dtype} {image.shape}')
        return image

    @staticmethod
    def _landmarks(landmarks):
        pts = np.asarray(landmarks, np.float64)
        if pts.size == 0:
            return pts.reshape(0, 5, 2)
        if pts.ndim == 2 and pts.shape[1] == 10:
            pts = pts.reshape(-1, 5, 2)  # interleaved x0, y0, x1, y1, ...
        if pts.ndim != 3 or pts.shape[1:] != (5, 2):
            raise ValueError(f'landmarks must be (N, 5, 2) or (N, 10), got {pts.shape}')
        if not np.isfinite(pts).all():
            raise ValueError('landmarks contain NaN/inf')
        return pts

    @staticmethod
    def _check_boxes(bboxes, pts):
        boxes = np.asarray(bboxes, np.float64)
        boxes = boxes.reshape(len(boxes), -1) if boxes.size else boxes.reshape(0, 4)
        if len(boxes) != len(pts):
            raise ValueError(f'{len(boxes)} bboxes but {len(pts)} landmark sets')
        if len(boxes) == 0:
            return
        if boxes.shape[1] < 4:
            raise ValueError(f'bboxes must be (N, 4+) x1, y1, x2, y2, got {boxes.shape}')
        x1, y1, x2, y2 = (boxes[:, j:j + 1] for j in range(4))
        w, h = x2 - x1, y2 - y1
        far = ((pts[..., 0] < x1 - w / 2) | (pts[..., 0] > x2 + w / 2) |
               (pts[..., 1] < y1 - h / 2) | (pts[..., 1] > y2 + h / 2)).any(1) | (w[:, 0] <= 0) | (h[:, 0] <= 0)
        if far.any():
            warnings.warn(f'{int(far.sum())} face(s): landmarks far outside their bbox (or invalid box) - expected '
                          f'boxes x1, y1, x2, y2 and landmarks (x, y) in pixels of the same image', stacklevel=3)
