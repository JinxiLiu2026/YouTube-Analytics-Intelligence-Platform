from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.dimension_vectors import DimensionVectorEncoder, signature_similarity


def print_signature(label: str, title: str, tags: list[str], signature) -> None:
    print(label)
    print("Title:", title)
    print("Tags:", ", ".join(tags))
    print("Phrase units:")
    for phrase in signature.phrase_units:
        print("-", phrase)
    print("Phrase projections:")
    for proj in signature.projections:
        print(
            f"- [{proj.source}] {proj.phrase}: "
            f"format={proj.format_weight:.3f} "
            f"context={proj.context_weight:.3f} "
            f"style={proj.style_weight:.3f}"
        )
    print()


def run_pair(encoder, left_label: str, left_example: dict, right_label: str, right_example: dict) -> None:
    sig_left = encoder.encode_video(left_example["title"], left_example["tags"])
    sig_right = encoder.encode_video(right_example["title"], right_example["tags"])
    sim = signature_similarity(sig_left, sig_right)

    print_signature(f"=== {left_label} ===", left_example["title"], left_example["tags"], sig_left)
    print_signature(f"=== {right_label} ===", right_example["title"], right_example["tags"], sig_right)
    print(
        f"{left_label} vs {right_label}:",
        f"topic={sim['topic']:.4f}",
        f"format={sim['format']:.4f}",
        f"context={sim['context']:.4f}",
        f"style={sim['style']:.4f}",
        f"total={sim['total']:.4f}",
    )
    print()


def main() -> None:
    example_a = {
        "title": "I Tried Surviving 24 Hours in Tokyo's Weirdest Capsule Hotel",
        "tags": ["tokyo travel", "capsule hotel", "japan vlog", "24 hour challenge"],
    }

    example_b = {
        "title": "24 Hours in Osaka's Tiny Capsule Hotel Vlog",
        "tags": ["osaka", "travel", "capsule hotel", "japan", "vlog", "hotel challenge"],
    }

    example_c = {
        "title": "Official Tokyo Music Video Teaser",
        "tags": ["tokyo", "japan", "music video", "official teaser"],
    }

    example_d = {
        "title": "Mortal Kombat 2 Trailer Reaction and Breakdown",
        "tags": ["mortal kombat 2", "trailer reaction", "movie breakdown", "johnny cage"],
    }

    example_e = {
        "title": "Official Mortal Kombat 2 Trailer Teaser",
        "tags": ["mortal kombat 2", "official trailer", "movie teaser", "action movie"],
    }

    example_f = {
        "title": "Celebrity Interview Backstage at Tokyo Film Festival",
        "tags": ["celebrity interview", "backstage", "tokyo", "film festival"],
    }

    example_g = {
        "title": "My Cozy Kyoto Travel Vlog and Cafe Diary",
        "tags": ["kyoto vlog", "japan travel", "cafe diary", "cozy vlog"],
    }

    encoder = DimensionVectorEncoder()
    print("=== Dimension Vector Demo ===\n")
    print("=== Pair 1: Travel Similar vs Travel Different ===\n")
    run_pair(encoder, "Example A", example_a, "Example B", example_b)
    run_pair(encoder, "Example A", example_a, "Example C", example_c)

    print("=== Pair 2: Trailer Reaction vs Trailer Teaser ===\n")
    run_pair(encoder, "Example D", example_d, "Example E", example_e)

    print("=== Pair 3: Travel Vlog vs Interview Event ===\n")
    run_pair(encoder, "Example A", example_a, "Example F", example_f)

    print("=== Pair 4: Travel Vlog vs Different Travel Vlog Tone ===\n")
    run_pair(encoder, "Example A", example_a, "Example G", example_g)


if __name__ == "__main__":
    main()
