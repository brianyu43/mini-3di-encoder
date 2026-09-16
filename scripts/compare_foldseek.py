"""Compare this encoder's full PDB chain with a Foldseek-generated 3Di FASTA."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mini3di_encoder.atoms import read_backbone
from mini3di_encoder.chain import compare_sequence, encode_chain


def read_single_fasta(path: Path) -> str:
    lines = path.read_text().splitlines()
    headers = [line for line in lines if line.startswith(">")]
    if len(headers) != 1:
        raise ValueError("Expected exactly one FASTA record")
    sequence = "".join(line.strip() for line in lines if line and not line.startswith(">"))
    if not sequence:
        raise ValueError("Empty FASTA sequence")
    return sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--chain", default="A")
    parser.add_argument("--foldseek-fasta", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    encoded = encode_chain(read_backbone(args.structure, args.chain))
    reference = read_single_fasta(args.foldseek_fasta)
    comparison = compare_sequence(encoded.letters, reference)
    report = {
        "structure": str(args.structure),
        "chain": args.chain,
        "observed": encoded.letters,
        "reference": reference,
        "valid_count": sum(encoded.valid),
        "invalid_count": len(encoded.valid) - sum(encoded.valid),
        "invalid_positions": [
            {"index": i, "reason": reason}
            for i, (valid, reason) in enumerate(
                zip(encoded.valid, encoded.invalid_reasons, strict=True)
            )
            if not valid
        ],
        "comparison": asdict(comparison),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        if args.out.exists():
            raise FileExistsError(args.out)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    print(rendered, end="")
    raise SystemExit(0 if comparison.equal else 1)


if __name__ == "__main__":
    main()
