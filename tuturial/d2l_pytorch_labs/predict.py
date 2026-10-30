"""图像单样本预测或英文影评情感预测。"""
import argparse
import json
from labkit.inference import predict_image, predict_text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--device", default="auto")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="本地图片路径，类别限于 CIFAR-10 的十类")
    group.add_argument("--text", help="英文影评")
    args = p.parse_args()
    result = (predict_image(args.run_dir, args.image, args.device) if args.image
              else predict_text(args.run_dir, args.text, args.device))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
