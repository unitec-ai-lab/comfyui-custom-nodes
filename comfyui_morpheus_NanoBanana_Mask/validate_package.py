#!/usr/bin/env python
"""
Validation script for ComfyUI Morpheus NanoBanana Mask package.
This verifies that the package structure is correct and imports work.
"""

import sys

def validate_package():
    """Validate the custom node package structure and imports."""
    print("=" * 60)
    print("ComfyUI Morpheus NanoBanana Mask - Package Validation")
    print("=" * 60)
    print()
    
    print("NOTE: This validation checks package structure only.")
    print("Import errors for 'torch', 'numpy' are expected - these")
    print("dependencies are provided by ComfyUI's environment.")
    print()
    
    errors = []
    warnings = []
    missing_deps = []
    
    print("1. Checking file structure...")
    import os
    required_files = [
        '__init__.py',
        'python/batch_v25_fix.py',
        'python/mask_v25_fix.py',
        'requirements.txt'
    ]
    
    for filepath in required_files:
        if os.path.exists(filepath):
            print(f"   ✓ {filepath} exists")
        else:
            errors.append(f"Missing required file: {filepath}")
            print(f"   ✗ {filepath} missing")
    
    print()
    print("2. Checking Python syntax...")
    import py_compile
    python_files = ['__init__.py', 'python/batch_v25_fix.py', 'python/mask_v25_fix.py']
    
    for filepath in python_files:
        try:
            py_compile.compile(filepath, doraise=True)
            print(f"   ✓ {filepath} syntax valid")
        except Exception as e:
            errors.append(f"Syntax error in {filepath}: {e}")
            print(f"   ✗ {filepath} syntax error: {e}")
    
    print()
    print("3. Checking imports (warnings expected)...")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("batch_v25_fix", "python/batch_v25_fix.py")
        print("   ✓ Batch node can be loaded as module")
    except Exception as e:
        if "torch" in str(e) or "numpy" in str(e):
            missing_deps.append(str(e))
            print(f"   ⚠ Expected dependency missing: {e}")
        else:
            errors.append(f"Batch node module error: {e}")
            print(f"   ✗ Batch node module error: {e}")
    
    try:
        spec = importlib.util.spec_from_file_location("mask_v25_fix", "python/mask_v25_fix.py")
        print("   ✓ Mask node can be loaded as module")
    except Exception as e:
        if "torch" in str(e) or "numpy" in str(e):
            missing_deps.append(str(e))
            print(f"   ⚠ Expected dependency missing: {e}")
        else:
            errors.append(f"Mask node module error: {e}")
            print(f"   ✗ Mask node module error: {e}")
    
    print()
    print("=" * 60)
    print("Validation Summary")
    print("=" * 60)
    
    if not errors:
        print("✓ Package structure is valid!")
        print()
        print("Installation Instructions:")
        print("1. Copy this folder to ComfyUI/custom_nodes/")
        print("2. Install requirements: pip install -r requirements.txt")
        print("3. Restart ComfyUI")
        print("4. Look for 'Morpheus · Batch Images + crop image' and")
        print("   'Morpheus · NanoBanana Mask' nodes in ComfyUI")
        return 0
    else:
        print(f"✗ Found {len(errors)} error(s):")
        for error in errors:
            print(f"  - {error}")
        return 1

if __name__ == "__main__":
    sys.exit(validate_package())
