"""Evaluate held-out labeled images, or classify an image, folder or video."""
import argparse
import json
from pathlib import Path
from classification_utils import PROJECT_ROOT, require_classifier, validate_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Path to trained classification best.pt')
    parser.add_argument('--data', default=str(PROJECT_ROOT / 'data' / 'civet_classification'))
    parser.add_argument('--split', choices=('test', 'val'), default='test')
    parser.add_argument('--source', help='Image, folder, or video; when set, predict instead of evaluating')
    parser.add_argument('--imgsz', type=int, default=224)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default=None)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--project', default=str(PROJECT_ROOT / 'runs' / 'classify'))
    parser.add_argument('--name', default=None)
    parser.add_argument('--show', action='store_true', help='Display predictions in a window')
    args = parser.parse_args()
    if not Path(args.model).is_file():
        parser.error(f'Model weights do not exist: {args.model}')
    if min(args.imgsz, args.batch) < 1 or args.workers < 0:
        parser.error('imgsz and batch must be positive; workers nonnegative.')
    if args.source and not Path(args.source).exists():
        parser.error(f'Source does not exist: {args.source}')
    if not args.source:
        try:
            data, classes = validate_dataset(args.data, (args.split,))
        except ValueError as error:
            parser.error(str(error))
    from ultralytics import YOLO
    model = YOLO(args.model)
    require_classifier(model)
    if args.source:
        results = model.predict(source=args.source, imgsz=args.imgsz, device=args.device,
                                stream=True, save=True, show=args.show,
                                project=args.project, name=args.name or 'predict', exist_ok=False)
        for index, result in enumerate(results):
            probs = result.probs
            print(f'{index}: {result.path}: {result.names[probs.top1]} '
                  f'({float(probs.top1conf):.4f})')
        return
    model_classes = [model.names[i] for i in range(len(model.names))]
    if classes != model_classes:
        parser.error(f'Dataset classes {classes} do not match model class order {model_classes}.')
    metrics = model.val(data=str(data), split=args.split, imgsz=args.imgsz,
                        batch=args.batch, device=args.device, workers=args.workers,
                        plots=True, project=args.project, name=args.name or args.split,
                        exist_ok=False)
    summary = {'split': args.split, 'top1_accuracy': float(metrics.top1),
               'top5_accuracy': float(metrics.top5), 'classes': classes,
               'model': str(Path(args.model).resolve()), 'data': str(data)}
    output = Path(model.validator.save_dir) / 'evaluation.json'
    output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'Evaluation and plots: {output.parent}')


if __name__ == '__main__':
    main()
