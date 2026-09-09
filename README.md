# SolRadViewer

SolRadViewer brings solar radio images, context images, and dynamic spectra into one workspace. It runs locally in your browser, with a shared clock for comparing observations in space and time.

I built it to study the spatial and temporal relationship between supra-arcade downflows and radio sources observed by EOVSA. That work involved comparing faint moving structures in EUV images with radio emission at different frequencies. The app brings those comparisons together: you can follow an event, adjust the image processing, and measure motion along a slit without switching between separate plots.

![SolRadViewer workspace with a dynamic spectrum, time–distance map, and layered image panels](https://raw.githubusercontent.com/sageyu123/solradviewer/main/docs/images/solradviewer-workspace.png)

*AIA 131 Å images and EOVSA observations of the 2025-03-28 event. The curved slit connects the image measurements to the time–distance map above.*

## Features

### Images as layers

Two image panels let you compare different views of the same event. Each panel holds image and contour layers with their own display settings. Reorder layers, toggle their visibility, or copy a layer to the other panel. Use sliders to adjust layer opacity and intensity limits as you inspect the images. Overlay radio contours on a context image, adjust the contour levels, and compare radio frequencies. Alignment controls shift the radio overlay when the observations need a positional correction.

### A shared timeline and dynamic spectrum

The dynamic spectrum shows radio intensity across time and frequency. Click or drag across it to select a time, scrub with the time slider, step through frames, or play the sequence. Time indicators connect the spectrum and image views; image timestamps and time offsets show which observations are being compared when the source cadences differ. Frequency controls select the radio channels to inspect.

### Image processing

Switch between intensity, difference, and ratio images. Use a previous frame, a fixed base frame, or a mean image as the reference to bring out changes over time. Each image layer has its own colormap, display range, and linear, logarithmic, square-root, or asinh stretch. A radial filter enhances coronal structure, and temporal low-pass or band-pass filters are available for difference and ratio images. Keep the original image in the other panel to see what the processing changes.

### Time–distance maps

Draw a straight or curved slit on an image and extract intensity along it through time. Adjust the slit width and smoothing, or draw a fan of slits to compare nearby paths. The resulting maps share the event timeline and can be exported as PNG images or NPZ arrays. Bind slits to different layers to compare motion in context images with radio emission, and adjust time shifts when inspecting their relationship.

You can also inspect pixel light curves, select regions for feature tracking, and extract radio peak and centroid positions. Save the workspace as a JSON session to return to the same setup, or export measurements as CSV files.

## Install and run

Requires Python 3.10–3.12. Install from [PyPI](https://pypi.org/project/solradviewer/) in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install solradviewer
solradviewer
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

Open [http://127.0.0.1:8010](http://127.0.0.1:8010). The package includes the browser interface; no separate frontend installation is needed. Press `Ctrl+C` in the terminal to stop the server. To choose another port, run `solradviewer --port 8020`.

## Load observations

Start with an example manifest for the [2022-01-18 flare](https://github.com/sageyu123/solradviewer/blob/main/manifests/eovsa_20220118_mflare.json) or the [2025-03-28 event](https://github.com/sageyu123/solradviewer/blob/main/manifests/ovro_lwa_20250328_cme.json). A manifest is a JSON file that lists the data sources and their locations on your computer. Replace the example paths with your own, then choose **Load Manifest / JSON** in the Data panel or drag the file into the workspace. Saved sessions load the same way.

The Data panel includes a local file browser. Its **Add Source Path** action currently registers a placeholder; use a manifest to load the supported image sequences. Observation files stay on the machine running the app. The example manifests reference data you supply; the observations are not included in the package.

### Supported data

| Data | Supported input |
| --- | --- |
| Context image sequences | AIA-style 2-D FITS images with solar coordinates and observation times, or HDF5 map sequences with per-image metadata |
| Radio image sequences | EOVSA all-band FITS files containing a frequency cube and channel metadata for each time |
| Dynamic spectra | EOVSA FITS spectra with frequency and time tables |
| Dataset descriptions and saved workspaces | JSON manifests and sessions |

A session currently needs both a context image sequence and a radio image sequence; the dynamic spectrum is optional. FITS support depends on the layouts above. Direct Helioviewer downloads and JPEG 2000 (JP2/JP2K) loading are not implemented in this release.

See the [data guide](https://github.com/sageyu123/solradviewer/blob/main/docs/data-and-configuration.md) for the required FITS and HDF5 layouts, a complete manifest example, and data/cache settings.

## Development

For an editable installation, frontend setup, and tests, see the [development guide](https://github.com/sageyu123/solradviewer/blob/main/docs/development.md). The backend is Python/FastAPI and the interface is React/TypeScript. Packaging and publishing are covered in the [release guide](https://github.com/sageyu123/solradviewer/blob/main/docs/releasing.md).
