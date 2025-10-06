import os
import subprocess
from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as ET

def _clean_musicxml(xml_file: str) -> None:
    """Remove unwanted annotations (harmony, repeats, endings) from a MusicXML file."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        tags_to_remove = {"harmony", "repeat", "ending"}

        for parent in root.iter():
            for child in list(parent):
                local_tag = child.tag.split('}')[-1]
                if local_tag in tags_to_remove:
                    parent.remove(child)

        tree.write(xml_file, encoding="utf-8", xml_declaration=True)
    except Exception as exc:
        print(f"Warning: Failed to clean annotations from {xml_file}: {exc}")


def convert_to_musicxml(score_path: str, xml_path: str):
    """Convert *score_path* (.mscx) to uncompressed MusicXML and clean it."""
    try:
        subprocess.run(['mscore', score_path, '-o', xml_path], check=True, capture_output=True)
        _clean_musicxml(xml_path)
    except Exception as e:
        print(f"Conversion with MuseScore failed for {os.path.basename(score_path)}: {e}")
        # Remove partial output on failure
        if os.path.exists(xml_path):
            os.remove(xml_path)
        return


def main():
    score_root = "data_root/dcml_unified/MS3"
    xml_root = "data_root/dcml_unified/musicxml"
    os.makedirs(xml_root, exist_ok=True)
    for score_path in tqdm(Path(score_root).glob("*.mscx")):
        xml_path = Path(xml_root) / score_path.name.replace(".mscx", ".musicxml")
        convert_to_musicxml(score_path, xml_path)


if __name__ == "__main__":
    main()