"""Train YOLO11 classification on labeled civet crops."""
import argparse
from classification_utils import PROJECT_ROOT, require_classifier, validate_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=str(PROJECT_ROOT / 'data' / 'civet_classification'))
    parser.add_argument('--model', default='yolo11n-cls.pt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=224)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--device', default=None, help='0 for NVIDIA GPU, cpu for CPU; default: automatic')
    parser.add_argument('--workers', type=int, default=0, help='Start at 0 for Windows compatibility')
    parser.add_argument('--project', default=str(PROJECT_ROOT / 'models'))
    parser.add_argument('--name', default='civet_classifier_yolo11n')
    args = parser.parse_args()
    if min(args.epochs, args.imgsz, args.batch) < 1 or min(args.workers, args.patience) < 0:
        parser.error('epochs, imgsz and batch must be positive; workers and patience nonnegative.')
    try:
        data, classes = validate_dataset(args.data, ('valid',))
    except ValueError as error:
        parser.error(str(error))
    from ultralytics import YOLO
    model = YOLO(args.model)
    require_classifier(model)
    print(f'Classes: {classes}')
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, patience=args.patience, device=args.device,
                workers=args.workers, project=args.project, name=args.name,
                seed=42, plots=True, save=True, cache=False, exist_ok=False)
    print(f'Best weights: {model.trainer.best}')


if __name__ == '__main__':
    main()
