"""
Regenerate the per-camera ``report.html`` inside existing route sessions.

Use this after changing the report template (e.g. adding KPI cards or the
frame-by-frame detections explorer) so that sessions recorded before the change
render with the current template. It re-renders from artifacts already on disk
(``summary.json`` + ``detections.json``) and never re-runs a model.

The model key and video name come from ``session_manifest.json`` -- attributing
every report to a hardcoded model would silently mislabel sessions run with a
different one. Sessions whose manifest is missing or unreadable are SKIPPED
rather than guessed at.

    python scripts/update_session_reports.py --dry-run     # show what would change
    python scripts/update_session_reports.py               # write
    python scripts/update_session_reports.py --session RamKundh_CMM
"""

import argparse
import json
import os
import sys

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure src is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pipeline.html_report import export_html_report  # noqa: E402


def _load_json(path: str):
    """Return parsed JSON, or None if absent/unreadable (reason printed)."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] Could not read {os.path.relpath(path, PROJECT_ROOT)}: {e}")
        return None


def _report_model_key(manifest: dict) -> str | None:
    """Model whose report.html the live pipeline would have left on disk.

    ``_execute_camera_pipeline`` loops over ``req.models`` and writes
    ``report.html`` once per model, so the LAST model in the list is the one
    that survives. Mirror that rather than picking the first.
    """
    models = manifest.get("models") or []
    models = [str(m) for m in models if m]
    return models[-1] if models else None


def update_sessions(only_session: str | None = None, dry_run: bool = False) -> int:
    sessions_dir = os.path.join(PROJECT_ROOT, "outputs", "sessions")
    if not os.path.isdir(sessions_dir):
        print(f"No sessions directory at {sessions_dir}")
        return 0

    updated_count = 0
    skipped_count = 0

    for sess_name in sorted(os.listdir(sessions_dir)):
        if only_session and sess_name != only_session:
            continue
        sess_path = os.path.join(sessions_dir, sess_name)
        if not os.path.isdir(sess_path):
            continue

        print(f"\n{sess_name}")

        manifest = _load_json(os.path.join(sess_path, "session_manifest.json"))
        if not isinstance(manifest, dict):
            print("  [SKIP] no readable session_manifest.json -- cannot determine "
                  "which model produced these runs.")
            skipped_count += 1
            continue

        model_key = _report_model_key(manifest)
        if not model_key:
            print("  [SKIP] manifest lists no models -- cannot attribute the report.")
            skipped_count += 1
            continue

        cameras = manifest.get("cameras") or {}
        if not isinstance(cameras, dict) or not cameras:
            print("  [SKIP] manifest lists no cameras.")
            skipped_count += 1
            continue

        for cam_id, cam in sorted(cameras.items()):
            cam = cam if isinstance(cam, dict) else {}
            # run_dir in the manifest is absolute and from the machine that
            # recorded the session; resolve by camera id under this session.
            cam_dir = os.path.join(sess_path, cam_id)
            if not os.path.isdir(cam_dir):
                print(f"  [skip] {cam_id}: no directory on disk")
                continue

            summary = _load_json(os.path.join(cam_dir, "summary.json"))
            if not isinstance(summary, dict):
                print(f"  [skip] {cam_id}: no readable summary.json "
                      f"(camera status={cam.get('status', '?')})")
                continue

            # export_html_report handles both dicts and Detection objects.
            detections = _load_json(os.path.join(cam_dir, "detections.json"))
            if detections is not None and not isinstance(detections, list):
                print(f"  [skip] {cam_id}: detections.json is not a list")
                detections = None

            video_name = (
                summary.get("video_name")
                or cam.get("video_name")
                or f"{cam_id}.mp4"
            )
            report_file = os.path.join(cam_dir, "report.html")
            verb = "would rewrite" if dry_run else "rewrote"
            exists = "overwriting" if os.path.isfile(report_file) else "creating"

            if not dry_run:
                try:
                    export_html_report(
                        output_path=report_file,
                        video_name=video_name,
                        model_key=model_key,
                        summary=summary,
                        detections=detections,
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"  [FAIL] {cam_id}: {e.__class__.__name__}: {e}")
                    continue

            print(f"  [OK] {verb} {cam_id}/report.html ({exists}, model={model_key}, "
                  f"{len(detections or [])} detections)")
            updated_count += 1

    noun = "would be regenerated" if dry_run else "regenerated"
    print(f"\nDone. {updated_count} camera sub-reports {noun}"
          + (f", {skipped_count} session(s) skipped." if skipped_count else "."))
    if dry_run and updated_count:
        print("Re-run without --dry-run to write.")
    return updated_count


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", help="Only update this session directory name.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List the reports that would be rewritten; write nothing.")
    args = ap.parse_args()
    update_sessions(only_session=args.session, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
