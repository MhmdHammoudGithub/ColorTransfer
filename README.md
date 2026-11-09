# Color Transfer with Sliced Wasserstein Projections

A notebook-based image-processing project that transfers a reference image's color distribution to a source image. It compares statistical color matching with an iterative optimal-transport-inspired method and explores how transport-map regularization affects visual artifacts.

Implemented with **Python, NumPy, PyTorch, scikit-image and OpenCV**, with Matplotlib and Plotly visualizations.

## Methods

### Mean and covariance matching

An affine transform aligns the source and target RGB statistics. The implementation computes covariance eigendecompositions, whitens the source colors and applies the target covariance and mean. It includes regularization for nearly singular covariance matrices and prints the resulting statistics before clipping to the displayable range.

### Per-channel histogram matching

Each source channel is mapped to the target channel's distribution through sorting and quantile interpolation. The implementation supports **RGB and Lab** and handles images with different numbers of pixels.



### Sliced Wasserstein projections

The PyTorch implementation repeatedly:

1. Samples random directions in RGB space.
2. Projects source and target colors onto each direction.
3. Matches the sorted one-dimensional distributions, interpolating target quantiles when image sizes differ.
4. Aggregates displacement vectors and updates the source colors.

The notebooks expose the number of iterations, directions per iteration, update step size and stopping thresholds. They record a sampled projection discrepancy and compare step sizes through convergence plots. Computation can run on a CUDA GPU or the CPU.

### Transportation-map regularization

An experimental postprocessing stage smooths the color-displacement field using OpenCV's **joint bilateral filter**, guided by the source image. It iteratively updates pixels according to a change threshold and visualizes intermediate results and failure cases.

## Repository contents

| File or directory | Contents |
| --- | --- |
| [`Color_Transfer_using_mean_and_covariance_matching.ipynb`](Color_Transfer_using_mean_and_covariance_matching.ipynb) | Affine RGB color transfer and statistical comparisons |
| [`Color_Transfer_using_histogram_matching_per_channel.ipynb`](Color_Transfer_using_histogram_matching_per_channel.ipynb) | RGB/Lab histogram matching |
| [`Color_Transfer_using_SWP.ipynb`](Color_Transfer_using_SWP.ipynb) | Sliced projections, stopping criteria and step-size sweeps |
| [`artefacts_TMR + color_transfer_applications + bad_results.ipynb`](artefacts_TMR%20%2B%20color_transfer_applications%20%2B%20bad_results.ipynb) | Regularization, example applications and unsuccessful transfers |
| [`Test Images/`](Test%20Images/) | Example source and target images |

## Installation

Create and activate a Python environment, then install the notebook dependencies:

```bash
python -m pip install numpy matplotlib plotly scikit-image opencv-contrib-python tqdm jupyterlab
```

Install PyTorch using the [official installation selector](https://pytorch.org/get-started/locally/) for your platform and compute device.

Use **`opencv-contrib-python`**, which includes `cv2.ximgproc.jointBilateralFilter`. Install only one OpenCV wheel variant in the environment, as explained in the [package documentation](https://pypi.org/project/opencv-contrib-python/).

## Running the notebooks

Launch Jupyter from the repository root:

```bash
jupyter lab
```

Open the notebook for the method you want to explore and execute its definition cells before running the examples.

### Image paths

Some cells use paths such as `ColabFiles/source.jpg` or `bird1.jpg`. Update these to the supplied image directory, for example:

```python
from skimage import io

source = io.imread("Test Images/source.jpg")
target = io.imread("Test Images/target.jpg")
```

Use three-channel RGB images. The SWP function expects RGB values in the **0–255 range** and returns a `uint8` image with the source image's dimensions.

### Sliced projection example

After executing the function definitions in the SWP notebook:

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
result, curve = sliced_wasserstein_projection(
    source,
    target,
    n_iter=100,
    n_dirs=10,
    device=device,
)
visualize_images_and_distributions(source, target, result)
```

This is an example configuration. The notebooks contain additional parameter settings and sweeps. Their existing calls often specify `device='cuda'`; change those calls when running without a CUDA GPU.

### Color ordering for regularization

`skimage.io.imread` loads RGB images, while `tmr_iterative` uses OpenCV **BGR** conversion constants. Convert explicitly when passing an RGB transfer result to this function:

```python
import cv2

source_bgr = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
regularized_bgr = tmr_iterative(source_bgr, result_bgr)
regularized_rgb = cv2.cvtColor(regularized_bgr, cv2.COLOR_BGR2RGB)
```

Run the regularization notebook's function-definition cells first. The original source and transferred image must have the same shape. Some existing demonstration cells pass RGB arrays directly, so apply this conversion when reproducing those examples.

## Outputs and interpretation

The notebooks display source/target/result comparisons, color-distribution plots and iteration curves. The SWP parameter-sweep helper also writes `tau_sweep.pdf` in the working directory.

- Color matching changes the palette; it does not establish semantic correspondences between objects in the two images.
- Random projection directions make SWP runs stochastic. Set NumPy/PyTorch seeds when comparing configurations.
- A small change in the sampled discrepancy is a stopping heuristic. Reaching an iteration limit does not establish convergence or visual quality.
- Regularization is experimental, and the notebooks retain examples where the transfer produces artifacts.
