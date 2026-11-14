"""CLI execution entrypoint to generate a product description using pretrained FLAN-T5."""

import argparse
import json
import os
import sys

# Ensure offline mode and src on path
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from descra.models.baseline import BaselineGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate product description using baseline FLAN-T5.")
    parser.add_argument(
        "--product-name",
        type=str,
        default="Vitamin C Brightening Facial Serum",
        help="Product title.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="Beauty & Personal Care",
        help="Product category.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=[
            "Infused with 20% pure Vitamin C and Hyaluronic Acid",
            "Targets dark spots, fine lines, and uneven skin tone",
            "Non-greasy, lightweight texture suitable for sensitive skin",
        ],
        help="List of product bullet features.",
    )
    parser.add_argument(
        "--specs",
        type=str,
        default='{"Volume": "30 ml", "Skin Type": "All Skin Types", "Item Form": "Serum"}',
        help="JSON string of technical specifications.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="beam_search",
        choices=["beam_search", "greedy"],
        help="Decoding strategy.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device ('auto', 'cuda', 'cpu').",
    )
    args = parser.parse_args()

    try:
        specs_dict = json.loads(args.specs)
    except Exception:
        specs_dict = {}

    print("=" * 60)
    print("DESCRA — BASELINE FLAN-T5 INFERENCE ENGINE")
    print("=" * 60)

    generator = BaselineGenerator(
        model_name="google/flan-t5-small",
        device=args.device,
        use_fp16=True,
    )

    print("\n[Input Product Metadata]")
    print(f"Name:       {args.product_name}")
    print(f"Category:   {args.category}")
    print("Features:")
    for f in args.features:
        print(f"  - {f}")
    print("Specifications:")
    for k, v in specs_dict.items():
        print(f"  - {k}: {v}")

    print(f"\n[Generating Description using '{args.strategy}' strategy]...")
    result = generator.generate(
        product_name=args.product_name,
        category=args.category,
        features=args.features,
        specifications=specs_dict,
        strategy=args.strategy,
    )

    print("\n" + "=" * 60)
    print("GENERATED PRODUCT DESCRIPTION:")
    print("=" * 60)
    print(result["description"])
    print("=" * 60)
    meta = result["metadata"]
    print(f"Latency:        {meta['latency_ms']:.2f} ms")
    print(f"Throughput:     {meta['tokens_per_second']:.2f} tokens/sec")
    print(f"Input Tokens:   {meta['input_tokens']}")
    print(f"Output Tokens:  {meta['output_tokens']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

