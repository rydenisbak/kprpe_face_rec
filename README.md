# kprpe_face — standalone KP-RPE ViT-B face descriptor

**English** | [Русский](README-ru.md)

One class, `FaceRecognizer`: an image + boxes + 5 landmarks of every face → 512-d descriptors.
The model is KP-RPE ViT-B (CVLface, trained on WebFace12M); the code is reduced to inference only and verified
bit-exact against the original.

> **The weights are for non-commercial research only.** They are not part of this repository and are downloaded
> from the authors' Hugging Face repository on first use. The weights have no license of their own; the model card
> asks to follow the license of the training data (WebFace260M / WebFace12M: academic research only, no commercial
> use). See [Licenses](#licenses).

## Installation
```bash
pip install git+https://github.com/rydenisbak/kprpe_face_rec.git
# or from a clone: pip install .   (pip install -e . for development)
```
Dependencies: numpy, opencv-python, torch, safetensors.

The weights (460 MB) are downloaded automatically on the first `FaceRecognizer()` into `~/.cache/kprpe_face/`
(pinned revision of the Hugging Face repository; size and sha256 are checked). To download them in advance, e.g.
when building a Docker image:
```bash
kprpe-face-download
```
Without pip: `python -m kprpe_face.download`. Environment variables: `KPRPE_FACE_CACHE` — cache folder,
`KPRPE_FACE_URL` — a mirror of the same file. Offline: put the file into `kprpe_face/weights/model.safetensors` or
into the cache, or pass the path with `FaceRecognizer(weights=...)`; `download=False` disables downloading.

## Usage
```python
import cv2
from kprpe_face import FaceRecognizer

rec = FaceRecognizer()                                   # device: cuda > mps > cpu
image = cv2.cvtColor(cv2.imread('frame.jpg'), cv2.COLOR_BGR2RGB)
descr = rec(image, bboxes, landmarks)                    # (N, 512) float32, L2-normalized
descr, norms = rec(image, bboxes, landmarks, return_norms=True)
sim = descr @ descr.T                                    # cosine similarity
```

| input | format |
|---|---|
| `image` | HxWx3 uint8, RGB; for BGR use `FaceRecognizer(input_color='bgr')` |
| `landmarks` | (N, 5, 2) in pixels of this image, (x, y): left eye, right eye, nose, left mouth corner, right mouth corner ("left" = image left, as in RetinaFace / InsightFace); or (N, 10) interleaved x0, y0, x1, y1, … |
| `bboxes` | (N, 4+) x1, y1, x2, y2[, score] or `None`. The model does not need the boxes (alignment uses only the landmarks); they are checked for count and for the landmarks lying near their box, otherwise a warning is issued (usually a mixed-up format) |

- All faces of one image go through the network as one batch (split into chunks of `batch_size=64` only if there
  are more faces).
- `norms` — the norm of the raw embedding; it grows with face quality. To aggregate a track, average the
  descriptors weighted by the norm.
- `flip=True` — flip test-time augmentation (sum of the embeddings of the crop and its mirror), about 2× slower.
- Debugging: `rec.align(image, landmarks)` → 112×112 RGB crops and landmarks in [0, 1];
  `rec.embed_aligned(crops, keypoints)` → raw embeddings.

## How it works
1. Similarity transform (shift, rotation, uniform scale) from the 5 landmarks onto the ArcFace 112×112 template,
   least squares (Umeyama); `cv2.warpAffine`, bilinear, black outside the image.
2. The same 5 landmarks are mapped into the crop with the same matrix and divided by 112 → the second input of the
   network.
3. RGB, (x/255 − 0.5)/0.5 → ViT-B: 8×8 patches → 196 tokens, width 512, 24 blocks, 16 heads, 512-d output.
   The landmarks affect only the attention: offsets from each patch to the 5 landmarks → a linear layer → for every
   block and head, a table of additive attention-logit biases over 7×7 cells of the neighbour's relative position.
4. L2 normalization.

## Verification
Compared with the original CVLface code (ViT + KP-RPE) on a private set of 2903 real faces (selfies and camera
frames):
- all 283 weight tensors and the RPE cell index match;
- aligned crops and landmarks are bit-identical;
- with the same batching (64 faces per batch, MPS) the descriptors are bit-identical, with and without flip;
- with a different batch size or device there is only float32 noise: max per-component difference 2.7e-7
  (batch of 1, MPS) and 1.3e-6 (CPU vs MPS), 1 − cos ≤ 3e-11;
- an image with 8 faces in one batch is bit-identical to calling face by face; BGR input and (N, 10) landmarks give
  the same result.

## Speed (Apple M1 Max)
- MPS: ~20 ms per call with one face (including alignment); in batches of 64, ~10 ms per face.
- CPU: ~65 ms per face (batch of 64).

## Licenses
- The code of this repository is MIT (`LICENSE`). It includes adapted code from CVLface (MIT), Microsoft iRPE (MIT),
  InsightFace (MIT) and scikit-image (BSD-3-Clause); their notices are in `THIRD_PARTY_NOTICES.md`.
- The weights are not part of this repository and are not covered by these licenses. The CVLface authors publish
  them on Hugging Face (`minchul/cvlface_adaface_vit_base_kprpe_webface12m`) without a license of their own and ask to
  follow the license of the training data: WebFace260M and its subsets are for academic research only, commercial use
  is prohibited. Therefore this package with these weights is for non-commercial research only.
