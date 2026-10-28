
import argparse
import json
import matplotlib.pyplot as plt
import visualtorch
from src.model import SmallPatchCNN


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the image-branch CNN.")
    parser.add_argument("--config", default="processed_dataset/config.json")
    parser.add_argument("--output", default="cnn_architecture.png")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        config = json.load(file)
    channels, height, width = config["patch_shape"]
    model = SmallPatchCNN(in_channels=channels, out_dim=128)
    image = visualtorch.layered_view(
        model,
        input_shape=(1, channels, height, width),
        spacing=10,
        scale_z=2.0,
        scale_xy=10.0,
        min_z=30,
        min_xy=30,
        max_z=100,
        max_xy=200,
        draw_volume=True,
    )

    plt.figure(figsize=(12, 4))
    plt.axis("off")
    plt.imshow(image)
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")
    plt.close()
    print(f"Saved architecture diagram to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
