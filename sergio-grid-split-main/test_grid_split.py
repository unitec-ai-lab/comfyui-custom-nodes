"""
Test script for GridSplitAuto, GridAnalyze, and GridSplitByCell nodes

This script tests the grid splitting functionality without requiring ComfyUI.
It creates test grid images and verifies the splitting works correctly.
"""

import numpy as np
from PIL import Image
from grid_split_auto import GridSplitAuto, NODE_CLASS_MAPPINGS as AUTO_MAP
from grid_analyze import GridAnalyze, NODE_CLASS_MAPPINGS as ANALYZE_MAP
from grid_split_by_cell import GridSplitByCell, NODE_CLASS_MAPPINGS as CELL_MAP


def create_test_grid_image(rows: int, cols: int, cell_size: int = 100, border: int = 2) -> np.ndarray:
    """
    Create a test grid image with distinct colored cells.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        cell_size: Size of each cell in pixels
        border: Border width between cells
        
    Returns:
        NumPy array of the test image (H, W, C) normalized to 0-1 range
    """
    height = rows * cell_size + (rows - 1) * border
    width = cols * cell_size + (cols - 1) * border
    
    img = np.ones((height, width, 3), dtype=np.float32) * 0.2
    
    colors = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.5, 0.0],
        [0.5, 0.0, 1.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.0],
    ]
    
    cell_idx = 0
    for r in range(rows):
        for c in range(cols):
            y0 = r * (cell_size + border)
            x0 = c * (cell_size + border)
            y1 = y0 + cell_size
            x1 = x0 + cell_size
            
            color = colors[cell_idx % len(colors)]
            img[y0:y1, x0:x1] = color
            cell_idx += 1
    
    return img


def test_manual_mode():
    """Test manual mode splitting."""
    print("=" * 50)
    print("Test: Manual Mode (3x3 grid)")
    print("=" * 50)
    
    test_img = create_test_grid_image(3, 3, cell_size=100, border=4)
    print(f"Created test image: {test_img.shape}")
    
    node = GridSplitAuto()
    
    batch, meta = node.split_grid(
        image=test_img,
        mode="manual",
        rows=3,
        cols=3,
        inset_px=2,
        min_cell_size=50,
        return_order="row_major"
    )
    
    print(f"Output batch shape: {batch.shape}")
    print(f"Number of cells: {batch.shape[0]}")
    print(f"\nMetadata:\n{meta}")
    
    assert batch.shape[0] == 9, f"Expected 9 cells, got {batch.shape[0]}"
    print("\nPASSED: Manual mode test")
    return True


def test_auto_mode():
    """Test auto mode detection."""
    print("\n" + "=" * 50)
    print("Test: Auto Mode (2x2 grid)")
    print("=" * 50)
    
    test_img = create_test_grid_image(2, 2, cell_size=150, border=6)
    print(f"Created test image: {test_img.shape}")
    
    node = GridSplitAuto()
    
    batch, meta = node.split_grid(
        image=test_img,
        mode="auto",
        rows=1,
        cols=1,
        inset_px=3,
        min_cell_size=50,
        return_order="row_major"
    )
    
    print(f"Output batch shape: {batch.shape}")
    print(f"Number of cells: {batch.shape[0]}")
    print(f"\nMetadata:\n{meta}")
    
    assert batch.shape[0] == 4, f"Expected 4 cells, got {batch.shape[0]}"
    print("\nPASSED: Auto mode test")
    return True


def test_column_major_order():
    """Test column-major ordering."""
    print("\n" + "=" * 50)
    print("Test: Column Major Order (2x3 grid)")
    print("=" * 50)
    
    test_img = create_test_grid_image(2, 3, cell_size=100, border=4)
    print(f"Created test image: {test_img.shape}")
    
    node = GridSplitAuto()
    
    batch_row, _ = node.split_grid(
        image=test_img,
        mode="manual",
        rows=2,
        cols=3,
        inset_px=2,
        min_cell_size=50,
        return_order="row_major"
    )
    
    batch_col, meta = node.split_grid(
        image=test_img,
        mode="manual",
        rows=2,
        cols=3,
        inset_px=2,
        min_cell_size=50,
        return_order="column_major"
    )
    
    print(f"Row-major batch shape: {batch_row.shape}")
    print(f"Column-major batch shape: {batch_col.shape}")
    print(f"\nMetadata:\n{meta}")
    
    assert batch_row.shape[0] == 6, f"Expected 6 cells, got {batch_row.shape[0]}"
    assert batch_col.shape[0] == 6, f"Expected 6 cells, got {batch_col.shape[0]}"
    
    assert not np.allclose(batch_row[1], batch_col[1]), "Orders should differ"
    
    print("\nPASSED: Column major order test")
    return True


def test_large_inset_error():
    """Test that large inset values are handled correctly."""
    print("\n" + "=" * 50)
    print("Test: Large Inset Handling")
    print("=" * 50)
    
    test_img = create_test_grid_image(2, 2, cell_size=50, border=2)
    print(f"Created test image: {test_img.shape}")
    
    node = GridSplitAuto()
    
    try:
        batch, meta = node.split_grid(
            image=test_img,
            mode="manual",
            rows=2,
            cols=2,
            inset_px=30,
            min_cell_size=20,
            return_order="row_major"
        )
        print("No error raised - inset was acceptable")
        print(f"Output batch shape: {batch.shape}")
    except ValueError as e:
        print(f"Expected error raised: {e}")
    
    print("\nPASSED: Large inset handling test")
    return True


def test_5x2_grid():
    """Test 5x2 grid layout."""
    print("\n" + "=" * 50)
    print("Test: 5x2 Grid Layout")
    print("=" * 50)
    
    test_img = create_test_grid_image(5, 2, cell_size=80, border=3)
    print(f"Created test image: {test_img.shape}")
    
    node = GridSplitAuto()
    
    batch, meta = node.split_grid(
        image=test_img,
        mode="manual",
        rows=5,
        cols=2,
        inset_px=1,
        min_cell_size=50,
        return_order="row_major"
    )
    
    print(f"Output batch shape: {batch.shape}")
    print(f"Number of cells: {batch.shape[0]}")
    print(f"\nMetadata:\n{meta}")
    
    assert batch.shape[0] == 10, f"Expected 10 cells, got {batch.shape[0]}"
    print("\nPASSED: 5x2 grid test")
    return True


def test_grid_analyze():
    """Test GridAnalyze node."""
    print("\n" + "=" * 50)
    print("Test: GridAnalyze Node (3x3 grid)")
    print("=" * 50)
    
    test_img = create_test_grid_image(3, 3, cell_size=100, border=4)
    print(f"Created test image: {test_img.shape}")
    
    node = GridAnalyze()
    
    preview, cell_width, cell_height, rows, cols, meta = node.analyze_grid(
        image=test_img,
        detection_mode="by_cell_count",
        rows_hint=3,
        cols_hint=3,
        cell_width_hint=0,
        cell_height_hint=0,
        preview_line_width=3
    )
    
    print(f"Preview shape: {preview.shape}")
    print(f"Detected: {rows}x{cols} grid")
    print(f"Cell size: {cell_width}x{cell_height}")
    print(f"\nMetadata:\n{meta}")
    
    assert rows == 3, f"Expected 3 rows, got {rows}"
    assert cols == 3, f"Expected 3 cols, got {cols}"
    
    preview_uint8 = (preview[0] * 255).astype(np.uint8)
    pil_preview = Image.fromarray(preview_uint8)
    pil_preview.save("test_grid_analyze_preview.png")
    print("Saved: test_grid_analyze_preview.png")
    
    print("\nPASSED: GridAnalyze test")
    return True


def test_grid_analyze_auto():
    """Test GridAnalyze node with auto detection."""
    print("\n" + "=" * 50)
    print("Test: GridAnalyze Auto Detection (2x2 grid)")
    print("=" * 50)
    
    test_img = create_test_grid_image(2, 2, cell_size=120, border=5)
    print(f"Created test image: {test_img.shape}")
    
    node = GridAnalyze()
    
    preview, cell_width, cell_height, rows, cols, meta = node.analyze_grid(
        image=test_img,
        detection_mode="auto",
        rows_hint=1,
        cols_hint=1,
        cell_width_hint=0,
        cell_height_hint=0,
        preview_line_width=3
    )
    
    print(f"Preview shape: {preview.shape}")
    print(f"Detected: {rows}x{cols} grid")
    print(f"Cell size: {cell_width}x{cell_height}")
    print(f"\nMetadata:\n{meta}")
    
    assert rows == 2, f"Expected 2 rows, got {rows}"
    assert cols == 2, f"Expected 2 cols, got {cols}"
    
    print("\nPASSED: GridAnalyze auto test")
    return True


def test_grid_split_by_cell():
    """Test GridSplitByCell node."""
    print("\n" + "=" * 50)
    print("Test: GridSplitByCell Node (3x3 grid)")
    print("=" * 50)
    
    test_img = create_test_grid_image(3, 3, cell_size=100, border=4)
    height, width = test_img.shape[:2]
    print(f"Created test image: {test_img.shape}")
    
    cell_width = width // 3
    cell_height = height // 3
    print(f"Using cell size: {cell_width}x{cell_height}")
    
    node = GridSplitByCell()
    
    batch, meta = node.split_by_cell(
        image=test_img,
        cell_width=cell_width,
        cell_height=cell_height,
        inset_px=2,
        return_order="row_major",
        skip_partial_cells="yes"
    )
    
    print(f"Output batch shape: {batch.shape}")
    print(f"Number of cells: {batch.shape[0]}")
    print(f"\nMetadata:\n{meta}")
    
    assert batch.shape[0] == 9, f"Expected 9 cells, got {batch.shape[0]}"
    print("\nPASSED: GridSplitByCell test")
    return True


def test_workflow_integration():
    """Test the full workflow: GridAnalyze -> GridSplitByCell"""
    print("\n" + "=" * 50)
    print("Test: Full Workflow (Analyze -> Split)")
    print("=" * 50)
    
    test_img = create_test_grid_image(2, 4, cell_size=80, border=3)
    print(f"Created test image: {test_img.shape}")
    
    analyze_node = GridAnalyze()
    preview, cell_width, cell_height, rows, cols, analyze_meta = analyze_node.analyze_grid(
        image=test_img,
        detection_mode="by_cell_count",
        rows_hint=2,
        cols_hint=4,
        cell_width_hint=0,
        cell_height_hint=0,
        preview_line_width=2
    )
    
    print(f"Analyze detected: {rows}x{cols} grid, cell: {cell_width}x{cell_height}")
    
    split_node = GridSplitByCell()
    batch, split_meta = split_node.split_by_cell(
        image=test_img,
        cell_width=cell_width,
        cell_height=cell_height,
        inset_px=1,
        return_order="row_major",
        skip_partial_cells="yes"
    )
    
    print(f"Split produced: {batch.shape[0]} cells")
    print(f"\nAnalyze Meta:\n{analyze_meta}")
    print(f"\nSplit Meta:\n{split_meta}")
    
    assert batch.shape[0] == 8, f"Expected 8 cells, got {batch.shape[0]}"
    print("\nPASSED: Full workflow test")
    return True


def save_test_visualization():
    """Create and save a visualization of the grid splitting."""
    print("\n" + "=" * 50)
    print("Creating Test Visualization")
    print("=" * 50)
    
    test_img = create_test_grid_image(3, 3, cell_size=100, border=4)
    
    img_uint8 = (test_img * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    pil_img.save("test_grid_input.png")
    print("Saved: test_grid_input.png")
    
    node = GridSplitAuto()
    batch, meta = node.split_grid(
        image=test_img,
        mode="manual",
        rows=3,
        cols=3,
        inset_px=2,
        min_cell_size=50,
        return_order="row_major"
    )
    
    for i in range(min(batch.shape[0], 9)):
        cell = batch[i]
        cell_uint8 = (cell * 255).astype(np.uint8)
        pil_cell = Image.fromarray(cell_uint8)
        pil_cell.save(f"test_cell_{i}.png")
        print(f"Saved: test_cell_{i}.png")
    
    print(f"\nTotal cells saved: {batch.shape[0]}")
    print("Visualization complete!")


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# GridSplit Test Suite")
    print("#" * 60)
    
    tests = [
        test_manual_mode,
        test_auto_mode,
        test_column_major_order,
        test_large_inset_error,
        test_5x2_grid,
        test_grid_analyze,
        test_grid_analyze_auto,
        test_grid_split_by_cell,
        test_workflow_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"\nFAILED: {test.__name__}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "#" * 60)
    print(f"# Results: {passed} passed, {failed} failed")
    print("#" * 60)
    
    save_test_visualization()
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
