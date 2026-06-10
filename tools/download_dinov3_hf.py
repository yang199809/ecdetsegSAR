import argparse
import os

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--local-dir", default=os.path.join("weights", "dinov3-vits16-pretrain-lvd1689m"))
    parser.add_argument("--cache-dir", default=os.path.join("weights", "hf_cache"))
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_dir=args.local_dir,
    )
    print(path)


if __name__ == "__main__":
    main()
