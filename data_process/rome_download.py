import os
import shutil
import json
import subprocess
import requests
from tqdm import tqdm
import music21
import xml.etree.ElementTree as ET  # Added for MusicXML post-processing
from concurrent.futures import ProcessPoolExecutor, as_completed


def generate_new_name(path_str, corpus_root):
    relative_path = os.path.relpath(path_str, corpus_root)
    parts = relative_path.split(os.sep)
    if len(parts) > 1:
        composer_part = parts[1]
        if "," in composer_part:
            parts[1] = composer_part.split(",")[0]

    return "_".join(parts)


# ---------------------------------------------------------------------------
# MusicXML utilities
# ---------------------------------------------------------------------------

def _clean_musicxml(xml_file: str) -> None:
    """Remove unwanted annotations (harmony, repeats, endings) from a MusicXML file.

    The function edits the file in-place.
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        tags_to_remove = {"harmony", "repeat", "ending"}

        # Iterate through all parents and remove matching children
        for parent in root.iter():
            for child in list(parent):
                local_tag = child.tag.split('}')[-1]  # Strip namespace
                if local_tag in tags_to_remove:
                    parent.remove(child)

        tree.write(xml_file, encoding="utf-8", xml_declaration=True)
    except Exception as exc:
        # If cleaning fails, keep the original file but report the issue.
        print(f"Warning: Failed to clean annotations from {xml_file}: {exc}")


def convert_to_musicxml(score_path: str, xml_path: str):
    """Convert *score_path* to uncompressed MusicXML and clean annotations."""
    file_ext = os.path.splitext(score_path)[1].lower()
    try:
        if file_ext == '.krn':
            score = music21.converter.parse(score_path)
            score.write('musicxml', fp=xml_path)
        else:
            # Use MuseScore CLI for formats that music21 may not fully support.
            subprocess.run(['mscore', score_path, '-o', xml_path], check=True, capture_output=True)

        # Post-process MusicXML: strip annotations & repeats
        xml_path = xml_path if os.path.exists(xml_path) else xml_path.replace(".musicxml", ".xml")
        _clean_musicxml(xml_path)

    except Exception as e:
        backend = "music21" if file_ext == '.krn' else "MuseScore"
        raise RuntimeError(f"Conversion with {backend} failed for {os.path.basename(score_path)}") from e


def download_file(url, target_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")
        raise


def process_piece(piece_dir, output_root, corpus_root):
    new_name = generate_new_name(piece_dir, corpus_root)
    new_piece_path = os.path.join(output_root, new_name)
    os.makedirs(new_piece_path, exist_ok=True)

    shutil.copy(os.path.join(piece_dir, "analysis.txt"), new_piece_path)

    remote_json_path = os.path.join(piece_dir, "remote.json")
    if os.path.exists(remote_json_path):
        with open(remote_json_path) as f:
            remote_info = json.load(f)

        remote_urls = [
            v
            for k, v in remote_info.items()
            if "remote_score" in k and isinstance(v, str) and v.startswith("http")
        ]

        # sort remote_urls to prioritize mscx
        remote_urls.sort(key=lambda url: ".mscx" not in url)

        for url in remote_urls:
            score_filename = os.path.basename(url)
            temp_score_path = os.path.join(new_piece_path, score_filename)
            try:
                download_file(url, temp_score_path)
                xml_path = os.path.join(new_piece_path, f"{new_name}.musicxml")
                convert_to_musicxml(temp_score_path, xml_path)
                os.remove(temp_score_path)
                return
            except Exception as e:
                print(
                    f"Info: Failed to process remote score {url} for {new_name}. Reason: {e}"
                )
                if os.path.exists(temp_score_path):
                    os.remove(temp_score_path)
                continue

def main():
    corpus_root = "data_root/Corpus"
    output_root = "data_root/rome_flattened_mxl"

    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root)

    piece_dirs = []
    for root, _, files in os.walk(corpus_root):
        if "analysis.txt" in files and (
            "score.mxl" in files
            or "score.mid" in files
            or "remote.json" in files
            or "score.krn" in files
            or "score.mscx" in files
        ):
            piece_dirs.append(root)

    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(process_piece, piece_dir, output_root, corpus_root)
            for piece_dir in piece_dirs
        }
        for future in tqdm(
            as_completed(futures), total=len(piece_dirs), desc="Processing pieces"
        ):
            try:
                future.result()
            except Exception as e:
                print(f"A piece failed to process: {e}")


if __name__ == "__main__":
    main()
