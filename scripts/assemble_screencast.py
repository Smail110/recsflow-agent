"""Собрать MP4 из кадров интерфейса с паузами для чтения ответов."""

import argparse
import json
import math
import subprocess
from pathlib import Path


def frame_durations(frames, max_gap_seconds):
    """Use editorial reading pauses when supplied; otherwise retain capture gaps."""
    if not frames or not math.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError("Нужны кадры и положительный предел паузы")
    durations = []
    for index, frame in enumerate(frames):
        if "duration_seconds" in frame:
            seconds = float(frame["duration_seconds"])
        elif index + 1 < len(frames):
            seconds = min(max_gap_seconds, max(0.1, (frames[index + 1]["time"] - frame["time"]) / 1000))
        else:
            seconds = min(8.0, max_gap_seconds)
        if not math.isfinite(seconds) or not 0.04 <= seconds <= 30:
            raise ValueError("Длительность кадра должна быть от 0,04 до 30 секунд")
        durations.append(seconds)
    return durations


def main():
    import imageio_ffmpeg

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, default=root / "artifacts" / "screencast")
    parser.add_argument("--output", type=Path, default=root / "report" / "demo.mp4")
    parser.add_argument("--max-gap-seconds", type=float, default=12.0)
    args = parser.parse_args()
    folder = args.frames_dir.resolve()
    frames = json.loads((folder / "frames.json").read_text(encoding="utf-8"))
    lines = ["ffconcat version 1.0"]
    duration = 0.0
    durations = frame_durations(frames, args.max_gap_seconds)
    for frame, seconds in zip(frames, durations, strict=True):
        path = Path(frame["path"]).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file():
            raise ValueError("Кадр должен находиться в папке записи")
        duration += seconds
        escaped = path.as_posix().replace("'", "'\\''")
        lines.extend([f"file '{escaped}'", f"duration {seconds:.3f}"])
    lines.append(lines[-2])
    if not 120 <= duration <= 300:
        raise ValueError(f"Нужна запись от 2 до 5 минут, получено {duration:.1f} секунд")
    manifest = folder / "video.ffconcat"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "warning",
            "-safe",
            "0",
            "-f",
            "concat",
            "-i",
            str(manifest),
            "-vf",
            "fps=24",
            "-c:v",
            "libx264",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    # Concat repeats the last frame; the encoded container can be longer than
    # the requested frame timeline. Report the actual media duration.
    reader = imageio_ffmpeg.read_frames(str(output))
    try:
        media = next(reader)
    finally:
        reader.close()
    actual_duration = media["duration"]
    if not 120 <= actual_duration <= 300:
        raise ValueError(f"Длительность MP4 вне диапазона 2–5 минут: {actual_duration}")
    (output.parent / "video.json").write_text(
        json.dumps(
            {
                "frames": len(frames),
                "duration_seconds": actual_duration,
                "requested_timeline_seconds": round(duration, 3),
                "duration_source": "ffmpeg encoded-media metadata",
                "frames_directory": str(folder),
                "started_unix_ms": frames[0]["time"],
                "finished_unix_ms": frames[-1]["time"],
                "method": "Монтаж кадров реального интерфейса с явными паузами для чтения; без озвучки и подмены ответов.",
                "max_gap_seconds": args.max_gap_seconds,
                "frame_durations_seconds": durations,
                "bytes": output.stat().st_size,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Видео: {actual_duration:.1f} секунд, {output.stat().st_size / 1_000_000:.2f} МБ")


if __name__ == "__main__":
    main()
