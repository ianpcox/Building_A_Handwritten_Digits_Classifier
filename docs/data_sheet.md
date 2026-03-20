# Data Sheet: Digits Dataset (sklearn / optional MNIST)

## Dataset (primary: sklearn digits)

- **Name:** Digits dataset (load_digits) – 8×8 grayscale images of handwritten digits 0–9.
- **Source:** scikit-learn; originally from UCI (optdigits). Citation: Dua, D. and Graff, C. UCI Machine Learning Repository.
- **License:** Open for research and education.
- **Size:** 1,797 samples (64 features per image, 8×8 pixels).
- **Splits:** Train/test via stratified split (e.g. 80/20, random_state=42). No augmentation in baseline pipeline; optional augmentation (rotation, zoom) can be added.

## Optional: MNIST

- **Name:** MNIST – 28×28 grayscale digits.
- **Source:** LeCun et al.; available via keras/tensorflow or torchvision.
- **Note:** If the project is extended to full MNIST, document the version and split here; keep sklearn digits as an option for fast reproduction.

## Known biases and limitations

- **Resolution:** 8×8 is low; performance and human interpretability are high but not comparable to 28×28 MNIST.
- **Demographics:** Digit writers and collection methodology not documented in sklearn; assume mixed sources.
