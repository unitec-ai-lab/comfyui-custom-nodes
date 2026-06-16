# ComfyUI Grid Split

A ComfyUI custom node package for automatically detecting and splitting grid-based images (collages, contact sheets, storyboards) into individual cells.

## Features

- **Automatic Grid Detection**: Uses gradient-based projection profiles to detect grid dividers
- **Visual Preview**: See detected grid overlay before splitting
- **Modular Approach**: Define cell size and let the system calculate the grid
- **Manual Mode**: Specify exact rows and columns when needed
- **Flexible Inset**: Remove grid lines/borders with configurable pixel inset
- **Deterministic Output**: Consistent row-major or column-major ordering
- **No Heavy Dependencies**: Uses only NumPy and Pillow - no OpenCV or ML libraries

## Installation

1. Clone or copy this folder to your ComfyUI `custom_nodes` directory:
   ```
   ComfyUI/custom_nodes/comfyui-grid-split/
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Restart ComfyUI

## Nodes

### 1. GridAnalyze (Preview)

**Category**: `image/grid`

Analyzes an image to detect grid dimensions and outputs a preview with the grid overlay. Use this to verify detection before splitting.

#### Inputs

| Name | Type | Description |
|------|------|-------------|
| image | IMAGE | Input image containing a grid |
| detection_mode | COMBO | `auto`, `by_cell_count`, or `by_cell_size` |
| rows_hint | INT | Number of rows (used in by_cell_count mode) |
| cols_hint | INT | Number of columns (used in by_cell_count mode) |
| cell_width_hint | INT | Cell width in pixels (used in by_cell_size mode) |
| cell_height_hint | INT | Cell height in pixels (used in by_cell_size mode) |
| preview_line_width | INT | Width of preview overlay lines |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| preview | IMAGE | Image with grid overlay (green = first cell, red = grid lines) |
| cell_width | INT | Detected/calculated cell width |
| cell_height | INT | Detected/calculated cell height |
| rows | INT | Number of rows |
| cols | INT | Number of columns |
| meta | STRING | Debug info |

### 2. GridSplitByCell

**Category**: `image/grid`

Splits an image based on cell dimensions (the "module"). Connect to GridAnalyze outputs or specify manually.

#### Inputs

| Name | Type | Description |
|------|------|-------------|
| image | IMAGE | Input image containing a grid |
| cell_width | INT | Width of each cell in pixels |
| cell_height | INT | Height of each cell in pixels |
| inset_px | INT | Pixel inset applied to each crop |
| return_order | COMBO | `row_major` or `column_major` |
| skip_partial_cells | COMBO | Whether to skip cells that don't fit completely |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| images | IMAGE | Batch of cropped grid images |
| meta | STRING | Debug info |

### 3. GridSplitAuto (Original)

**Category**: `image/grid`

The original all-in-one node with auto and manual modes.

#### Inputs

| Name | Type | Description |
|------|------|-------------|
| image | IMAGE | Input image containing a grid |
| mode | COMBO | `auto` or `manual` |
| rows | INT | Number of rows (manual mode) |
| cols | INT | Number of columns (manual mode) |
| inset_px | INT | Pixel inset applied to each crop |
| min_cell_size | INT | Minimum allowed cell dimension |
| return_order | COMBO | `row_major` or `column_major` |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| images | IMAGE | Batch of cropped grid images |
| meta | STRING | Debug info |

## Recommended Workflows

### Workflow 1: Preview Before Split (Recommended)

```
Image → GridAnalyze (preview) → PreviewImage (verify grid)
                    ↓
              cell_width, cell_height
                    ↓
        GridSplitByCell → PreviewImage → ImageFromBatch → output
```

This workflow lets you:
1. See the detected grid overlay before splitting
2. Adjust cell dimensions if detection is wrong
3. Split with confidence

### Workflow 2: Direct Auto Split

```
Image → GridSplitAuto (auto mode) → PreviewImage → ImageFromBatch → output
```

For simpler grids where auto-detection usually works.

### Workflow 3: Manual Split

```
Image → GridSplitAuto (manual, rows=3, cols=3) → PreviewImage → output
```

When you know the exact grid dimensions.

## Grid Detection Algorithm

### Auto Mode Detection

1. **Preprocessing**: Convert to grayscale and compute gradient magnitude
2. **Projection Profiles**: Sum gradients per column (vertical) and row (horizontal)
3. **Divider Detection**: Find high-energy peaks indicating grid lines
4. **Fallback**: If detection fails, try common layouts (2×2, 3×3, 4×2, 5×2, etc.)

### Supported Grid Layouts

Common layouts detected automatically:
- 2×2, 3×3, 4×4
- 2×3, 3×2
- 2×4, 4×2
- 2×5, 5×2
- 3×4, 4×3
- 3×6, 6×3

## Troubleshooting

### Auto-detection produces wrong grid

Use **GridAnalyze** first to preview the detection:
1. Connect your image to GridAnalyze
2. Set `detection_mode` to `auto`
3. Look at the preview - the green rectangle shows the first cell
4. If wrong, switch to `by_cell_count` and specify rows/cols manually
5. Connect the cell_width/cell_height outputs to GridSplitByCell

### Crops include grid lines

Increase the `inset_px` value to crop away the grid lines from each cell.

### Error: Cell size too small

Either:
- Reduce `min_cell_size` parameter
- Use manual mode with fewer rows/cols

## Technology Stack

- Python 3.10+
- NumPy (gradient computation, array operations)
- Pillow/PIL (image handling, preview overlay)
- Native ComfyUI node API

No OpenCV or ML dependencies required.

## License

MIT License
