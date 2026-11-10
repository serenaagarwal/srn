"""Build anti-aliased HR/LR training pairs from original full-resolution images.

    original (e.g. 2040x1356) --/hr_downscale--> HR (~510x339) --/scale--> LR (~127x84)

Both steps use Pillow's BICUBIC resize, which low-pass filters before
subsampling (the equivalent of ImageJ's "Average when downsizing" being ON).
HR is trimmed to a multiple of `scale` first, so LR is exactly HR / scale and
the two grids line up with no offset.

Output layout matches what run_model.py expects:
    <out_root>/train/<name>_hr, <out_root>/train/<name>_lr
    <out_root>/val/<name>_hr,   <out_root>/val/<name>_lr

Example (Flickr2K: 000001-001999 train, 002000-002650 val, same as before):
    python make_sr_pairs.py --src /path/to/Flickr2K_HR --name flickr2 --val_start 2000
"""
import os
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def resize_bicubic(img, size):
    # Pillow's resize uses a support that widens with the downscale factor -> anti-aliased.
    return img.resize(size, Image.BICUBIC, reducing_gap=None)


def process_one(job):
    src_path, hr_path, lr_path, hr_downscale, scale = job
    img = Image.open(src_path).convert("RGB")
    w, h = img.size

    # 1) original -> HR
    hr_w, hr_h = round(w / hr_downscale), round(h / hr_downscale)
    # 2) trim to a multiple of scale by resizing the *original* to an exact size
    #    (keeps the aspect ratio within < 1 px; no crop offset to worry about)
    hr_w, hr_h = hr_w - hr_w % scale, hr_h - hr_h % scale
    hr = resize_bicubic(img, (hr_w, hr_h))

    # 3) HR -> LR, exactly HR / scale
    lr = resize_bicubic(hr, (hr_w // scale, hr_h // scale))

    hr.save(hr_path)
    lr.save(lr_path)

    # sanity metric: bicubic-upscaled LR vs HR (grayscale PSNR, same as training)
    hr_g = np.asarray(hr.convert("L"), dtype=np.float64) / 255
    up_g = np.asarray(lr.resize((hr_w, hr_h), Image.BICUBIC).convert("L"), dtype=np.float64) / 255
    mse = max(np.mean((hr_g - up_g) ** 2), 1e-10)
    return os.path.basename(hr_path), (hr_w, hr_h), 10 * np.log10(1 / mse)


def safe_process_one(job):
    # A truncated/corrupt download shouldn't kill the whole run; report it instead.
    # Both files are written only after both resizes succeed, so no half-written pairs.
    try:
        return process_one(job)
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="folder of original full-res images (not searched recursively)")
    p.add_argument("--out_root", default="dataset")
    p.add_argument("--name", default="flickr2", help="dataset prefix -> <name>_hr / <name>_lr")
    p.add_argument("--hr_downscale", type=float, default=4.0, help="original / HR size (default 4: 2K -> ~500px)")
    p.add_argument("--scale", type=int, default=4, help="HR / LR size (default 4)")
    p.add_argument("--val_start", type=int, default=2000,
                   help="images whose filename number (e.g. 002000.png -> 2000) is >= this go to val "
                        "(default 2000, Flickr2K split). Works even if some files are missing.")
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    names = sorted(f for f in os.listdir(args.src) if f.lower().endswith(IMG_EXTS))
    assert names, f"no images found in {args.src}"

    dirs = {}
    for split in ("train", "val"):
        for kind in ("hr", "lr"):
            d = os.path.join(args.out_root, split, f"{args.name}_{kind}")
            if os.path.isdir(d) and os.listdir(d) and not args.overwrite:
                raise SystemExit(f"{d} already has files; pass --overwrite or choose another --name")
            os.makedirs(d, exist_ok=True)
            dirs[split, kind] = d

    jobs = []
    for n in names:
        stem = os.path.splitext(n)[0]
        assert stem.isdigit(), f"expected numbered filenames like 000123.png, got {n}"
        split = "val" if int(stem) >= args.val_start else "train"
        out_name = os.path.splitext(n)[0] + ".png"  # always lossless PNG
        jobs.append((
            os.path.join(args.src, n),
            os.path.join(dirs[split, "hr"], out_name),
            os.path.join(dirs[split, "lr"], out_name),
            args.hr_downscale,
            args.scale,
        ))

    n_val = sum(1 for j in jobs if os.path.dirname(j[1]) == dirs["val", "hr"])
    print(f"{len(names)} images -> train {len(names) - n_val}, val {n_val} | workers={args.workers}", flush=True)
    assert n_val > 0 and n_val < len(names), "train/val split is empty on one side -- check --val_start"

    psnrs, failed = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, (job, res) in enumerate(zip(jobs, ex.map(safe_process_one, jobs, chunksize=8)), start=1):
            if isinstance(res, str):
                failed.append((os.path.basename(job[0]), res))
                print(f"SKIPPED {os.path.basename(job[0])}: {res}", flush=True)
                continue
            name, size, psnr = res
            psnrs.append(psnr)
            if k % 200 == 0 or k == len(jobs):
                print(f"[{k}/{len(jobs)}] {name} HR {size} | bicubic PSNR {psnr:.2f} dB", flush=True)

    if failed:
        print(f"{len(failed)} image(s) could not be read and were skipped: "
              f"{[f for f, _ in failed]}", flush=True)
    psnrs = np.array(psnrs)
    print(f"Done. Bicubic x{args.scale} PSNR over all pairs: mean {psnrs.mean():.2f} dB, "
          f"median {np.median(psnrs):.2f}, min {psnrs.min():.2f}", flush=True)


if __name__ == "__main__":
    main()
