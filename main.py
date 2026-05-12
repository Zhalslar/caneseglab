from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from caneseglab.config import ProjectConfig
from caneseglab.core import SegmentationProject


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CaneSegLab unified CLI entrypoint",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    mask_parser = subparsers.add_parser("mask", help="Convert labelme json to masks")
    mask_parser.add_argument("--input-dir", type=_path, required=True)
    mask_parser.add_argument("--output-dir", type=_path, default=None)

    train_parser = subparsers.add_parser("train", help="Train segmentation model")
    train_parser.add_argument("--image-dir", type=_path, required=True)
    train_parser.add_argument("--mask-dir", type=_path, required=True)

    export_parser = subparsers.add_parser("export-onnx", help="Export ONNX model")
    export_parser.add_argument("--output", type=_path, default=None)

    export_trt_parser = subparsers.add_parser(
        "export-trt",
        help="Build TensorRT engine from ONNX model",
    )
    export_trt_parser.add_argument("--onnx", type=_path, default=None)
    export_trt_parser.add_argument("--output", type=_path, default=None)

    verify_parser = subparsers.add_parser(
        "verify-onnx",
        help="Run ONNX inference and save visualization",
    )
    verify_parser.add_argument("--image", type=_path, default=None)

    nav_onnx_parser = subparsers.add_parser(
        "navigate-onnx",
        help="Run ONNX inference and fit navigation line",
    )
    nav_onnx_parser.add_argument("--image", type=_path, default=None)

    trt_parser = subparsers.add_parser(
        "infer-trt",
        help="Run TensorRT inference and save visualization",
    )
    trt_parser.add_argument("--image", type=_path, default=None)

    nav_trt_parser = subparsers.add_parser(
        "navigate-trt",
        help="Run TensorRT inference and fit navigation line",
    )
    nav_trt_parser.add_argument("--image", type=_path, default=None)

    nav_dataset_parser = subparsers.add_parser(
        "navigate-dataset",
        help="Run ONNX navigation fitting for a dataset",
    )
    nav_dataset_parser.add_argument("--image-dir", type=_path, required=True)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark inference backends on a dataset image directory",
    )
    benchmark_parser.add_argument("--image-dir", type=_path, required=True)
    benchmark_parser.add_argument("--output-dir", type=_path, default=None)
    benchmark_parser.add_argument("--backends", type=str, default=None)
    benchmark_parser.add_argument("--warmup-runs", type=int, default=None)
    benchmark_parser.add_argument("--timed-runs", type=int, default=None)
    benchmark_parser.add_argument("--max-images", type=int, default=None)

    audit_parser = subparsers.add_parser(
        "audit-navigation",
        help="Audit navigation JSON results and export failure reports",
    )
    audit_parser.add_argument("--navigation-root", type=_path, default=None)
    audit_parser.add_argument("--output-dir", type=_path, default=None)

    paper_parser = subparsers.add_parser(
        "export-paper-figures",
        help="Export paper-ready figures from training history and navigation results",
    )
    paper_parser.add_argument("--history", type=_path, default=None)
    paper_parser.add_argument("--navigation-root", type=_path, default=None)
    paper_parser.add_argument("--output-dir", type=_path, default=None)

    web_parser = subparsers.add_parser("web", help="Start aiohttp web console")
    web_parser.add_argument("--host", type=str, default=None)
    web_parser.add_argument("--port", type=int, default=None)
    web_parser.add_argument("--open-browser", action="store_true")

    return parser


def run(args: argparse.Namespace) -> int:
    config = ProjectConfig()
    project = SegmentationProject(config)

    if args.command == "mask":
        output_paths = project.generate_masks(args.input_dir, args.output_dir)
        print(f"转换完成: success={len(output_paths)}")
        return 0

    if args.command == "train":
        project.train(args.image_dir, args.mask_dir)
        print(f"训练完成，输出目录: {project.training_service.output_dir.resolve()}")
        return 0

    if args.command == "export-onnx":
        exported = project.export_onnx(args.output)
        print(f"导出成功: {exported}")
        return 0

    if args.command == "export-trt":
        exported = project.export_tensorrt(
            onnx_path=args.onnx,
            engine_output_path=args.output,
        )
        print(f"导出成功: {exported}")
        return 0

    if args.command == "verify-onnx":
        image_path = args.image or config.inference.image_path
        mask_output, overlay_output = project.verify_onnx(image_path)
        print(f"mask 已保存: {mask_output}")
        print(f"overlay 已保存: {overlay_output}")
        return 0

    if args.command == "infer-trt":
        image_path = args.image or config.inference.image_path
        mask_output, overlay_output = project.infer_tensorrt(image_path)
        print(f"mask 已保存: {mask_output}")
        print(f"overlay 已保存: {overlay_output}")
        return 0

    if args.command == "navigate-onnx":
        image_path = args.image or config.inference.image_path
        result = project.navigate_onnx(image_path)
        print(f"mask 已保存: {result['mask_path']}")
        print(f"overlay 已保存: {result['overlay_path']}")
        print(f"导航图已保存: {result['nav_overlay_output']}")
        print(f"导航数据已保存: {result['nav_json_output']}")
        return 0

    if args.command == "navigate-trt":
        image_path = args.image or config.inference.image_path
        result = project.navigate_tensorrt(image_path)
        print(f"mask 已保存: {result['mask_path']}")
        print(f"overlay 已保存: {result['overlay_path']}")
        print(f"导航图已保存: {result['nav_overlay_output']}")
        print(f"导航数据已保存: {result['nav_json_output']}")
        return 0

    if args.command == "navigate-dataset":
        result = project.navigate_dataset_onnx(args.image_dir)
        print(
            "导航批量处理完成: "
            f"processed={result['processed_count']}, "
            f"skipped={result['skipped_existing']}"
        )
        return 0

    if args.command == "benchmark":
        backends = [item.strip() for item in args.backends.split(",") if item.strip()] if args.backends else None
        result = project.benchmark_inference(
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            backends=backends,
            warmup_runs=args.warmup_runs,
            timed_runs=args.timed_runs,
            max_images=args.max_images,
        )
        print(f"测速结果: {result['summary_path']}")
        print(f"汇总表: {result['table_path']}")
        for item in result.get("results", []):
            if item.get("status") == "ok":
                print(
                    f"{item['backend']}: avg={item['avg_ms']} ms, "
                    f"p95={item['p95_ms']} ms, fps={item['fps']}"
                )
            else:
                print(f"{item['backend']}: failed - {item.get('error', 'unknown error')}")
        return 0

    if args.command == "audit-navigation":
        result = project.audit_navigation_results(
            navigation_root=args.navigation_root,
            output_dir=args.output_dir,
        )
        print(f"导航审计 JSON: {result['summary_path']}")
        print(f"失败样例 CSV: {result['failure_table_path']}")
        print(f"总导航结果: {result['total_nav_json']}")
        print(f"状态统计: {result['status_counts']}")
        return 0

    if args.command == "export-paper-figures":
        result = project.export_paper_figures(
            history_path=args.history,
            navigation_root=args.navigation_root,
            output_dir=args.output_dir,
        )
        print(f"论文图导出目录: {result['output_dir']}")
        for name, outputs in result.get("figures", {}).items():
            print(f"{name}:")
            for output_path in outputs:
                if isinstance(output_path, list):
                    for path in output_path:
                        print(f"  {path}")
                    continue
                print(f"  {output_path}")
        return 0

    if args.command == "web":
        from caneseglab.webapp import run_web_app

        run_web_app(
            config,
            host=args.host,
            port=args.port,
            open_browser=args.open_browser,
        )
        return 0

    raise ValueError(f"Unknown command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
