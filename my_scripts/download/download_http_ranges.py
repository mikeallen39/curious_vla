#!/usr/bin/env python3
import argparse
import concurrent.futures
import os
import sys
import time
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output")
    parser.add_argument("--parts", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--part-dir", default=None)
    return parser.parse_args()


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "curious-vla-repair/1.0"})
    return session


def resolve_size(session: requests.Session, url: str) -> int:
    response = session.head(url, allow_redirects=True, timeout=(30, 120))
    response.raise_for_status()
    length = response.headers.get("Content-Length")
    if length:
        return int(length)

    response = session.get(
        url,
        headers={"Range": "bytes=0-0"},
        allow_redirects=True,
        stream=True,
        timeout=(30, 120),
    )
    response.raise_for_status()
    content_range = response.headers.get("Content-Range", "")
    if "/" in content_range:
        return int(content_range.rsplit("/", 1)[1])
    raise RuntimeError(f"failed to resolve content length for {url}")


def expected_ranges(size: int, parts: int) -> list[tuple[int, int, int]]:
    chunk = (size + parts - 1) // parts
    ranges = []
    for idx in range(parts):
        start = idx * chunk
        if start >= size:
            break
        end = min(start + chunk - 1, size - 1)
        ranges.append((idx, start, end))
    return ranges


def download_part(
    url: str,
    part_path: Path,
    start: int,
    end: int,
    retries: int,
) -> None:
    expected = end - start + 1
    downloaded = part_path.stat().st_size if part_path.exists() else 0
    if downloaded > expected:
        part_path.unlink()
        downloaded = 0

    attempt = 0
    while downloaded < expected:
        attempt += 1
        if attempt > retries:
            raise RuntimeError(
                f"part {part_path.name} exceeded retries, expected={expected}, downloaded={downloaded}"
            )

        session = make_session()
        range_start = start + downloaded
        headers = {"Range": f"bytes={range_start}-{end}"}
        try:
            with session.get(
                url,
                headers=headers,
                allow_redirects=True,
                stream=True,
                timeout=(30, 120),
            ) as response:
                response.raise_for_status()
                if response.status_code not in (200, 206):
                    raise RuntimeError(
                        f"unexpected status {response.status_code} for {part_path.name}"
                    )
                if response.status_code == 200 and downloaded > 0:
                    raise RuntimeError(
                        f"range not honored for resumed part {part_path.name}"
                    )

                mode = "ab" if downloaded > 0 else "wb"
                with part_path.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)

        except Exception as exc:
            downloaded = part_path.stat().st_size if part_path.exists() else 0
            print(
                f"retry part={part_path.name} downloaded={downloaded}/{expected} attempt={attempt} error={exc}",
                flush=True,
            )
            time.sleep(min(30, attempt * 2))
            continue

        downloaded = part_path.stat().st_size if part_path.exists() else 0
        print(
            f"part_progress part={part_path.name} downloaded={downloaded}/{expected}",
            flush=True,
        )

    print(f"part_done part={part_path.name} size={expected}", flush=True)


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    part_dir = Path(args.part_dir) if args.part_dir else output.parent / f"{output.name}.parts"
    part_dir.mkdir(parents=True, exist_ok=True)

    session = make_session()
    size = resolve_size(session, args.url)
    print(f"resolved_size={size}", flush=True)

    ranges = expected_ranges(size, args.parts)
    print(
        f"parts={len(ranges)} concurrency={min(args.concurrency, len(ranges))}",
        flush=True,
    )

    futures = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(args.concurrency, len(ranges))
    ) as executor:
        for idx, start, end in ranges:
            part_path = part_dir / f"{output.name}.part{idx}"
            futures.append(
                executor.submit(
                    download_part,
                    args.url,
                    part_path,
                    start,
                    end,
                    args.retries,
                )
            )

        for future in concurrent.futures.as_completed(futures):
            future.result()

    if output.exists() and output.stat().st_size == size:
        print(f"assembled_exists={output}", flush=True)
        return 0

    tmp_output = output.with_suffix(output.suffix + ".tmp")
    if tmp_output.exists():
        tmp_output.unlink()

    with tmp_output.open("wb") as handle:
        for idx, _, _ in ranges:
            part_path = part_dir / f"{output.name}.part{idx}"
            handle.write(part_path.read_bytes())

    actual = tmp_output.stat().st_size
    if actual != size:
        raise RuntimeError(f"assembled size mismatch expected={size} actual={actual}")

    os.replace(tmp_output, output)
    print(f"assembled={output} size={size}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
