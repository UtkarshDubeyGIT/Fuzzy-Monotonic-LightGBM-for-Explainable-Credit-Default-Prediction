import json
import os
import base64
from pathlib import Path


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
    nb_path = '/workspaces/Credit-Risk-Analysis-and-Prediction-Framework/EDA/process1.ipynb'
    out_dir = '/workspaces/Credit-Risk-Analysis-and-Prediction-Framework/figures'
    n = extract_pngs_from_notebook(nb_path, out_dir)
    print(f'Extracted {n} PNG images to {out_dir}')
