import sys
import subprocess


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: run_serpent_with_timeout.py <mid_file> [timeout_seconds]", file=sys.stderr)
        sys.exit(2)

    midi_path: str = sys.argv[1]
    try:
        timeout_seconds: float = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    except Exception:
        timeout_seconds = 60.0

    try:
        subprocess.run(
            ["serpent64", "serpent_get_timestamp.srp", midi_path],
            check=True,
            timeout=timeout_seconds,
        )
    except Exception as exc:
        print(f"serpent failed or timed out: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

