# Operationalization: Digit Recognition API

## Architecture

```mermaid
flowchart LR
  Image[Digit Image] --> Preproc[Preprocess] --> CNN[CNN]
  CNN --> Digit[Digit + Confidence]
  Digit --> API[API]
  API --> Forms[Form Processing]
```

(Note: Current pipeline uses MLP on 8×8 digits; diagram shows target state with CNN for higher-resolution input.)

## Target user and value proposition

**Target users:** Developers building form-digit extraction, document pipelines, or accessibility tools that need digit + confidence from a small image crop.

**Value proposition:** Accept an image (or 64-dim vector for 8×8); return predicted digit (0–9) and confidence. Low latency and small footprint when using MLP or a small CNN.

**Deployment:** REST API: POST image (or base64); preprocess to 8×8 (or 28×28 if using MNIST); run model; return `{"digit": int, "confidence": float}`. Optionally persist for analytics. Scale via load balancer.

## Next steps

1. **Add FastAPI wrapper:** Single endpoint; load model at startup; accept image and return digit + confidence.
2. **Optional CNN and MNIST:** Replace MLP with a small CNN and train on MNIST for higher resolution and better generalization to form scans.
3. **Preprocessing contract:** Document input size and normalization (e.g. grayscale, resize to 8×8 or 28×28, scale to [0,1]) so clients can send compatible images.
