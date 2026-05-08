"""
Pitch-shift query evaluation + save (query clip, top-1 retrieved clip) per trial.

**Fixed mode:** one folder per shift in ``--shifts`` (legacy grid).

**Stochastic mode (default):** random pitch per trial in
``[--pitch-sem-min, --pitch-sem-max]`` (default ±6 semitones) to probe
frequency invariance, matching typical contrastive augmentation ranges.

  python evaluate_pitch_shift_pairs.py \\
    --mp3 "music/britney_spears/11. Toxic.mp3" \\
    --out-dir outputs/pitch_eval_toxic --query-sec 10 --n 30

  python evaluate_pitch_shift_pairs.py --pitch-mode fixed --shifts 0 2 5 ...

"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd

from contrastive_audio import embed_waveform, load_checkpoint
from retrieval_eval_common import (
    rank_ids,
    retrieved_listening_clip,
    slug,
    song_id_from_path,
    write_wav,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-dir", type=str, default="data/contrastive_db")
    ap.add_argument("--mp3", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default="outputs/pitch_shift_eval")
    ap.add_argument("--query-sec", type=float, default=10.0)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--pitch-mode",
        type=str,
        default="stochastic",
        choices=("stochastic", "fixed"),
        help="stochastic: random semitones per trial in [min,max]. fixed: one folder per --shifts.",
    )
    ap.add_argument(
        "--pitch-sem-min",
        type=float,
        default=-6.0,
        help="Stochastic mode: lower bound (semitones).",
    )
    ap.add_argument(
        "--pitch-sem-max",
        type=float,
        default=6.0,
        help="Stochastic mode: upper bound (semitones).",
    )
    ap.add_argument(
        "--shifts",
        type=float,
        nargs="+",
        default=[0.0, 2.0, 5.0],
        help="Fixed mode only: semitone shift per condition (e.g. 0 2 5).",
    )
    ap.add_argument(
        "--retrieved-listening",
        type=str,
        default="query_aligned",
        choices=("query_aligned", "fixed_anchor"),
        help="query_aligned: same crop offset as query in predicted file. "
        "fixed_anchor: --retrieved-anchor-sec.",
    )
    ap.add_argument(
        "--retrieved-anchor-sec",
        type=float,
        default=30.0,
        help="With fixed_anchor: start time (s) in predicted file for saved clip",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    db = (root / args.db_dir).resolve()
    mp3_path = (root / args.mp3).resolve() if not Path(args.mp3).is_absolute() else Path(args.mp3)
    out_root = (root / args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for p in [db / "encoder.pt", db / "gallery.npz", db / "manifest.csv"]:
        if not p.is_file():
            print(f"Missing {p}", file=sys.stderr)
            sys.exit(1)

    manifest = pd.read_csv(db / "manifest.csv")
    true_id = song_id_from_path(manifest, mp3_path, root)

    data = np.load(db / "gallery.npz", allow_pickle=True)
    Z = np.asarray(data["embeddings"], dtype=np.float32)
    g_ids = [str(x) for x in data["song_ids"].tolist()]
    data.close()

    if true_id not in g_ids:
        print(f"Song {true_id!r} not in gallery", file=sys.stderr)
        sys.exit(1)

    model, mel_cfg = load_checkpoint(db / "encoder.pt", device=args.device, embed_dim=None)
    sr = int(mel_cfg.sr)
    y_full, _ = librosa.load(str(mp3_path), sr=sr, mono=True)
    y_full = y_full.astype(np.float32)

    n = int(args.n)
    L = int(round(float(args.query_sec) * sr))
    hi = len(y_full) - L
    if hi < 1:
        print("Audio shorter than query window", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    starts = rng.integers(0, hi, size=n, dtype=np.int64)

    summary: dict[str, Any] = {
        "mp3": str(mp3_path),
        "true_song_id": true_id,
        "gallery_size": len(g_ids),
        "query_sec": args.query_sec,
        "n_queries": n,
        "seed": args.seed,
        "pitch_mode": args.pitch_mode,
        "retrieval_note": (
            "Gallery = one vector per song (full audio). Rank picks a song id, not a segment. "
            "Saved 'retrieved' WAV is a listening aid from the predicted file."
        ),
        "retrieved_listening": args.retrieved_listening,
        "conditions": [],
    }

    def run_condition(sub: Path, semis_per_trial: np.ndarray, label: str) -> None:
        sub.mkdir(parents=True, exist_ok=True)
        top1 = 0
        mrrs: list[float] = []
        meta_rows: list[dict] = []
        t_embed = 0.0
        for i in range(n):
            s0 = int(starts[i])
            q = y_full[s0 : s0 + L].astype(np.float32).copy()
            semis = float(semis_per_trial[i])
            if abs(semis) > 1e-6:
                q = librosa.effects.pitch_shift(q, sr=sr, n_steps=semis).astype(np.float32)
            t0 = time.perf_counter()
            zq = embed_waveform(model, q, mel_cfg, args.device)
            t_embed += time.perf_counter() - t0
            ranked = rank_ids(zq, Z, g_ids)
            pred = ranked[0]
            try:
                r = ranked.index(true_id) + 1
            except ValueError:
                r = len(g_ids) + 1
            mrrs.append(1.0 / r)
            ok = pred == true_id
            if ok:
                top1 += 1
            q_start_arg = int(s0) if args.retrieved_listening == "query_aligned" else None
            ret_audio, r_start = retrieved_listening_clip(
                manifest,
                pred,
                sr,
                L,
                query_start_sample=q_start_arg,
                anchor_sec=args.retrieved_anchor_sec,
            )
            qname = sub / f"pair_{i:03d}_query_pitch_{label}.wav"
            rname = sub / f"pair_{i:03d}_retrieved_top1_{slug(pred)}.wav"
            write_wav(qname, q, sr)
            write_wav(rname, ret_audio, sr)
            meta_rows.append(
                {
                    "pair_index": i,
                    "pitch_semitones": semis,
                    "query_crop_start_sample": int(s0),
                    "query_crop_start_sec": float(s0 / sr),
                    "predicted_song_id": pred,
                    "true_rank": int(r),
                    "top1_correct": bool(ok),
                    "retrieved_clip_start_sample": int(r_start),
                    "retrieved_clip_start_sec": float(r_start / sr),
                    "retrieved_same_time_as_query": bool(r_start == s0),
                }
            )
        cond = {
            "pitch_mode": args.pitch_mode,
            "folder": str(sub.relative_to(root)),
            "top1_acc": top1 / n,
            "mrr": float(np.mean(mrrs)),
            "mrr_std": float(np.std(mrrs)),
            "mean_embed_sec": float(t_embed / n),
        }
        if args.pitch_mode == "stochastic":
            cond["pitch_sem_min"] = float(args.pitch_sem_min)
            cond["pitch_sem_max"] = float(args.pitch_sem_max)
        summary["conditions"].append(cond)
        (sub / "trial_meta.json").write_text(json.dumps(meta_rows, indent=2))
        (sub / "condition_summary.json").write_text(json.dumps(cond, indent=2))

    if args.pitch_mode == "stochastic":
        lo, hi_sem = float(args.pitch_sem_min), float(args.pitch_sem_max)
        if lo >= hi_sem:
            print("--pitch-sem-min must be < --pitch-sem-max", file=sys.stderr)
            sys.exit(1)
        label = f"stoch_{str(lo).replace('.', 'p')}_to_{str(hi_sem).replace('.', 'p')}sem"
        label = label.replace("-", "n")
        sub = out_root / label
        sem_draws = rng.uniform(lo, hi_sem, size=n)
        summary["pitch_sem_min"] = lo
        summary["pitch_sem_max"] = hi_sem
        run_condition(sub, sem_draws, label)
    else:
        summary["shifts_semitones"] = list(args.shifts)
        for semis in args.shifts:
            semis = float(semis)
            label = f"shift_{str(semis).replace('.', '_')}sem"
            sub = out_root / label
            sem_const = np.full(n, semis, dtype=np.float64)
            run_condition(sub, sem_const, label)

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
