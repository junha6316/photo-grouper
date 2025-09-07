# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from pathlib import Path

# Get the directory containing this spec file
spec_root = Path(SPECPATH)

block_cipher = None

# Define data files to include
datas = [
    (str(spec_root / 'ui'), 'ui'),
    (str(spec_root / 'core'), 'core'),
    (str(spec_root / 'infra'), 'infra'),
    (str(spec_root / 'utils'), 'utils'),
]
excludes= []

# Hidden imports for PyTorch and other dependencies
hiddenimports = [
    'torch',
    'torchvision',
    'PIL',
    'PIL._tkinter_finder',
    'numpy',
    'numpy._core',
    'numpy._core.multiarray',
    'numpy._core.overrides',
    'numpy.core',
    'numpy.core.multiarray',
    'sklearn',
    'faiss',
    'networkx',
    'PySide6.QtCore',
    'PySide6.QtGui', 
    'PySide6.QtWidgets',
    'aiofiles',
    'watchdog',
    'seaborn',
]

# Exclude unnecessary modules to reduce size


a = Analysis(
    ['app.py'],
    pathex=[str(spec_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={
        # Optimize torch imports
        'torch': {
            'backends': ['cpu'],  # Only CPU backend
        },
    },
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
    # Additional optimization options
    optimize=0,  # Python bytecode optimization
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Photo Grouper',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,  # Strip symbols for size reduction
    upx=True,    # Enable UPX compression
    upx_exclude=[
        # Exclude DLLs that may cause issues with UPX
        'vcruntime140.dll',
        'msvcp140.dll',
        'ucrtbase.dll',
        'api-ms-win-*.dll',
        'Qt6Core.dll',  # Large Qt DLLs might compress poorly
        'Qt6Gui.dll',
        'Qt6Widgets.dll',
        'python*.dll',
        # PyTorch related libraries
        'torch_cpu.dll',
        'libtorch_cpu.dll',
        'mkl_*.dll',
        'libiomp*.dll',
    ],
    runtime_tmpdir=None,
    console=False,  # Set to True for debugging
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(spec_root.parent.parent / 'assets' / 'icon.ico') if sys.platform == 'win32' 
         else str(spec_root.parent.parent / 'assets' / 'icon.icns') if sys.platform == 'darwin'
         else None,
    # Additional size optimization options
    optimize=0,  # Bytecode optimization level
    noupx=False,  # Enable UPX (set to True to disable if issues occur)
)

# macOS app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name='Photo Grouper.app',
        icon=str(spec_root.parent.parent / 'assets' / 'icon.icns'),
        bundle_identifier='com.photogrouper.app',
        info_plist={
            'CFBundleDisplayName': 'Photo Grouper',
            'CFBundleVersion': '1.0.0',
            'CFBundleShortVersionString': '1.0.0',
            'NSHighResolutionCapable': True,
            'NSRequiresAquaSystemAppearance': False,
        },
    )
