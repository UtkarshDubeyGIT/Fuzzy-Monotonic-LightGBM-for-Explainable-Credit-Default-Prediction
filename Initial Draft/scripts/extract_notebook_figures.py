import json
import os
import base64
from pathlib import Path
import argparse


def extract_pngs_from_notebook(nb_path: str, out_dir: str) -> int:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    count = 0
    for ci, cell in enumerate(nb.get('cells', []), start=1):
        if cell.get('cell_type') != 'code':
            continue
        for oi, out in enumerate(cell.get('outputs', []), start=1):
            data = out.get('data') or {}
            if 'image/png' in data:
                b64 = data['image/png']
                if isinstance(b64, list):
                    b64 = ''.join(b64)
                try:
                    img_bytes = base64.b64decode(b64)
                except Exception:
                    # Skip if decoding fails
                    continue
                count += 1
                fname = f'process1_img_{count:03d}.png'
                fpath = out_path / fname
                with open(fpath, 'wb') as wf:
                    wf.write(img_bytes)
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract PNG figures from a Jupyter notebook.')
    parser.add_argument('--nb', dest='nb_path', required=False,
                        default='/workspaces/Credit-Risk-Analysis-and-Prediction-Framework/EDA/process1.ipynb',
                        help='Path to the .ipynb file')
    parser.add_argument('--out', dest='out_dir', required=False,
                        default='/workspaces/Credit-Risk-Analysis-and-Prediction-Framework/figures',
                        help='Output directory for PNG files')
    parser.add_argument('--prefix', dest='prefix', required=False, default='process1_img_',
                        help='Filename prefix for saved images')
    args = parser.parse_args()

    # Monkey-patch filename pattern via wrapper
    def extract_with_prefix(nb_path: str, out_dir: str, prefix: str) -> int:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        count = 0
        for ci, cell in enumerate(nb.get('cells', []), start=1):
            if cell.get('cell_type') != 'code':
                continue
            for oi, out in enumerate(cell.get('outputs', []), start=1):
                data = out.get('data') or {}
                if 'image/png' in data:
                    b64 = data['image/png']
                    if isinstance(b64, list):
                        b64 = ''.join(b64)
                    try:
                        img_bytes = base64.b64decode(b64)
                    except Exception:
                        continue
                    count += 1
                    fname = f'{prefix}{count:03d}.png'
                    fpath = out_path / fname
                    with open(fpath, 'wb') as wf:
                        wf.write(img_bytes)
        return count

    n = extract_with_prefix(args.nb_path, args.out_dir, args.prefix)
    print(f'Extracted {n} PNG images to {args.out_dir}')
