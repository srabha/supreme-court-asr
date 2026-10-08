#!/usr/bin/env python3
"""
Downloads transcripts (PDF), audio (Dropbox mp3) and optionally YouTube audio
for every hearing in the "SC Transcripts | ML Assignment Speech" CSV.

Usage:
    pip install requests mutagen         # required (mutagen reads mp3 durations)
    pip install imageio-ffmpeg           # only if ffmpeg is not on PATH (converts m4a -> mp3)
    pip install yt-dlp                   # only for --youtube (also needs ffmpeg)

    python scripts/download_data.py "<your_csv_file>.csv"
    python scripts/download_data.py "<your_csv_file>.csv" --youtube
    python scripts/download_data.py "<your_csv_file>.csv" --only transcripts   # or: mp3
    python scripts/download_data.py "<your_csv_file>.csv" --verify-only        # just check mp3 durations

Output: ./sc_hearings/{transcripts,mp3,youtube}/ ; re-running skips existing files
(existing .mp3 files that are not real audio, e.g. a saved Dropbox HTML page, are re-downloaded).
Some Dropbox links named *.mp3 actually serve MP4/AAC (m4a) audio; those are re-encoded to mp3
(128 kbps, original sample rate/channels) and the m4a is deleted unless --keep-m4a is given.
Each mp3's duration is compared with the CSV's "Hearing Duration(in Minutes)" column and the
result is written to sc_hearings/mp3_duration_check.csv.
Failures are listed at the end and written to failed_downloads.txt.
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from urllib.parse import urlparse

import time

import requests
import mutagen

OUT_DIR = "sc_hearings"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
# Dropbox serves its HTML preview page (even with dl=1) to browser User-Agents
DROPBOX_HEADERS = {"User-Agent": "curl/8.0"}
DURATION_COL = "Hearing Duration(in Minutes)"


def read_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            if row.get("Hearing Date"):
                rows.append(row)
    return rows


def dropbox_direct(url):
    # dl=0 returns an HTML preview page; dl=1 returns the raw file
    if re.search(r"[?&]dl=0", url):
        return re.sub(r"([?&])dl=0", r"\1dl=1", url)
    return url + ("&" if "?" in url else "?") + "dl=1"


def audio_ext(path):
    """Real container of a downloaded audio file: ".mp3", ".m4a" or None (e.g. a saved HTML page).
    Several Dropbox links named *.mp3 actually serve MP4/AAC audio."""
    try:
        with open(path, "rb") as f:
            head = f.read(12)
    except OSError:
        return None
    if head[:3] == b"ID3" or (len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return ".mp3"
    if head[4:8] == b"ftyp":
        return ".m4a"
    return None


def find_audio(base):
    """Existing valid audio file for this hearing (base + .mp3/.m4a), or None."""
    for ext in (".mp3", ".m4a"):
        if audio_ext(base + ext) == ext:
            return base + ext
    return None


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError("ffmpeg not found: install it or `pip install imageio-ffmpeg`")


def m4a_to_mp3(src, dest, keep_src=False):
    """Re-encode m4a (AAC) audio to mp3 at 128 kbps, matching the native mp3s in the dataset."""
    tmp = dest + ".part.mp3"
    r = subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", src, "-vn",
                        "-c:a", "libmp3lame", "-b:a", "128k", tmp])
    if r.returncode != 0 or audio_ext(tmp) != ".mp3":
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"ffmpeg failed to convert {src}")
    src_min, out_min = audio_minutes(src), audio_minutes(tmp)
    if src_min is None or out_min is None or abs(src_min - out_min) > 0.1:
        os.remove(tmp)
        raise RuntimeError(f"converted duration {out_min} min differs from source {src_min} min")
    os.replace(tmp, dest)
    if not keep_src:
        os.remove(src)
    return dest


def audio_minutes(path):
    try:
        return mutagen.File(path).info.length / 60
    except Exception:
        return None


def download_audio(url, base, retries=8):
    """Download the exact file behind `url` with resume on dropped connections.
    Saved as base + the extension matching its real container; returns that path."""
    tmp = base + ".part"
    for attempt in range(1, retries + 1):
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        headers = dict(DROPBOX_HEADERS)
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with requests.get(url, stream=True, timeout=60, headers=headers) as r:
                if r.status_code == 416:  # already have every byte
                    break
                r.raise_for_status()
                ctype = r.headers.get("Content-Type", "")
                if "html" in ctype.lower():
                    raise RuntimeError(f"expected audio but got Content-Type '{ctype}'")
                mode = "ab" if have and r.status_code == 206 else "wb"
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            break
        except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError,
                requests.exceptions.Timeout) as e:
            if attempt == retries:
                raise
            print(f"   connection dropped ({e.__class__.__name__}), resuming (attempt {attempt + 1})")
            time.sleep(min(5 * attempt, 30))
    ext = audio_ext(tmp)
    if ext is None:
        os.remove(tmp)
        raise RuntimeError("downloaded file is not mp3/m4a audio")
    os.replace(tmp, base + ext)
    return base + ext


def download(url, dest, kind=None):
    tmp = dest + ".part"
    if "dropbox.com" in url:
        headers = dict(DROPBOX_HEADERS)
    else:
        headers = dict(HEADERS)
        headers["Accept"] = "application/pdf,application/octet-stream,*/*"
    if "sci.gov.in" in url or "s3waas.gov.in" in url:
        headers["Referer"] = "https://www.sci.gov.in/"
    session = requests.Session()
    try:
        r = session.get(url, stream=True, timeout=60, allow_redirects=True, headers=headers)
    except requests.exceptions.SSLError:
        # some .gov.in sites have incomplete certificate chains that browsers tolerate
        print("   SSL error, retrying without certificate verification")
        requests.packages.urllib3.disable_warnings()
        r = session.get(url, stream=True, timeout=60, allow_redirects=True, headers=headers, verify=False)
    with r:
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        if kind == "pdf" and "pdf" not in ctype.lower() and "octet-stream" not in ctype.lower():
            raise RuntimeError(f"expected a PDF but got Content-Type '{ctype}'")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    os.replace(tmp, dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--youtube", action="store_true", help="also download YouTube audio as mp3")
    ap.add_argument("--only", choices=["transcripts", "mp3"], help="restrict to one kind")
    ap.add_argument("--verify-only", action="store_true", help="skip downloads, only check audio durations")
    ap.add_argument("--keep-m4a", action="store_true", help="keep the original m4a after converting to mp3")
    ap.add_argument("--tolerance", type=float, default=2.0,
                    help="allowed |mp3 duration - CSV duration| in minutes (default 2)")
    args = ap.parse_args()
    if args.verify_only:
        args.only = "mp3"

    rows = read_rows(args.csv_path)
    print(f"{len(rows)} hearings found")
    failed = []
    checks = []  # (mp3 file, csv minutes, actual minutes, status)

    for kind in ("transcripts", "mp3", "youtube"):
        os.makedirs(os.path.join(OUT_DIR, kind), exist_ok=True)

    for i, row in enumerate(rows, 1):
        date = row["Hearing Date"].replace(".", "-")
        base = f"{i:02d}_{date}"

        # e.g. "001_161132009_2023-12-12" (index + transcript PDF name without extension)
        pdf_name = row.get("Transcript PDF Name") or os.path.basename(
            urlparse(row.get("Transcript Link [Outdated]", "")).path)
        stem = os.path.splitext(pdf_name)[0] or "-".join(reversed(row["Hearing Date"].split(".")))
        mp3_base = f"{i:03d}_{stem}"

        # Transcript PDF: the full URL is in the "[Outdated]" column; the other column is just a filename
        pdf_url = row.get("Transcript Link [Outdated]", "")
        if pdf_url.startswith("http") and args.only in (None, "transcripts"):
            name = row.get("Transcript PDF Name") or os.path.basename(urlparse(pdf_url).path)
            dest = os.path.join(OUT_DIR, "transcripts", f"{base}_{name}")
            if not os.path.exists(dest):
                print(f"[pdf ] {base}")
                try:
                    download(pdf_url, dest, kind="pdf")
                except Exception as e:
                    failed.append((pdf_url, str(e)))
                    print(f"   FAILED: {e}", file=sys.stderr)

        mp3_url = row.get("mp3 format link", "")
        if mp3_url and args.only in (None, "mp3"):
            base_path = os.path.join(OUT_DIR, "mp3", mp3_base)
            dest = find_audio(base_path)
            if dest is None and not args.verify_only:
                stale = base_path + ".mp3"
                if os.path.exists(stale):  # e.g. a Dropbox HTML page saved by an older version
                    print(f"   removing {stale}: not audio")
                    os.remove(stale)
                print(f"[mp3 ] {mp3_base}")
                try:
                    dest = download_audio(dropbox_direct(mp3_url), base_path)
                except Exception as e:
                    failed.append((mp3_url, str(e)))
                    print(f"   FAILED: {e}", file=sys.stderr)
            if dest and dest.endswith(".m4a") and not args.verify_only:
                print(f"[conv] {os.path.basename(dest)} -> mp3")
                try:
                    dest = m4a_to_mp3(dest, base_path + ".mp3", keep_src=args.keep_m4a)
                except Exception as e:
                    failed.append((mp3_url, str(e)))
                    print(f"   FAILED: {e}", file=sys.stderr)

            try:
                expected = float(row.get(DURATION_COL, ""))
            except ValueError:
                expected = None
            actual = audio_minutes(dest) if dest else None
            if actual is None:
                status = "MISSING/INVALID"
            elif expected is None:
                status = "NO CSV DURATION"
            elif abs(actual - expected) <= args.tolerance:
                status = "OK"
            else:
                status = "MISMATCH"
            checks.append((os.path.basename(dest or base_path + ".mp3"), expected, actual, status))

        yt_url = row.get("Oral Hearing Link", "")
        if args.youtube and yt_url and args.only is None:
            print(f"[yt  ] {base}")
            r = subprocess.run(
                ["yt-dlp", "-x", "--audio-format", "mp3",
                 "-o", os.path.join(OUT_DIR, "youtube", base + ".%(ext)s"), yt_url])
            if r.returncode != 0:
                failed.append((yt_url, "yt-dlp failed"))

    if checks:
        report = os.path.join(OUT_DIR, "mp3_duration_check.csv")
        with open(report, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["mp3_file", "csv_minutes", "mp3_minutes", "diff_minutes", "status"])
            print(f"\nmp3 duration check (tolerance {args.tolerance:g} min):")
            for name, exp, act, status in checks:
                diff = act - exp if act is not None and exp is not None else None
                w.writerow([name, exp, f"{act:.2f}" if act is not None else "",
                            f"{diff:+.2f}" if diff is not None else "", status])
                print(f"  {status:15} {name:40} csv={exp if exp is not None else '?':>6} "
                      f"mp3={f'{act:.1f}' if act is not None else '?':>6}")
        bad = sum(1 for c in checks if c[3] != "OK")
        print(f"{len(checks) - bad}/{len(checks)} mp3 durations match the CSV; report: {report}")

    if failed:
        with open("failed_downloads.txt", "w") as f:
            for u, e in failed:
                f.write(f"{u}\t{e}\n")
        print(f"\n{len(failed)} failures, see failed_downloads.txt")
    print("Done.")


if __name__ == "__main__":
    main()
