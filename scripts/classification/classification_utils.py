"""Dataset checks shared by the classification commands."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def validate_dataset(root, splits):
    """Require matching, nonempty class folders; never substitute test for valid."""
    root = Path(root).resolve()
    expected = None
    for split in dict.fromkeys(('train', *splits)):
        folder = root / split
        if not folder.is_dir():
            raise ValueError(f'Missing {folder}. Expected dataset/split/class_name/images.')
        classes = sorted(p.name for p in folder.iterdir() if p.is_dir())
        if len(classes) < 2:
            raise ValueError(f'{folder} must contain at least two class folders.')
        if expected is not None and classes != expected:
            raise ValueError(f'Class folders in {folder} differ from train: {expected}')
        expected = classes
        for name in classes:
            if not any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                       for p in (folder / name).rglob('*')):
                raise ValueError(f'No supported images in {folder / name}')
    return root, expected


def require_classifier(model):
    if model.task != 'classify':
        raise ValueError('Use classification weights (-cls.pt or a trained classifier), not detector weights.')
