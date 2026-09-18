import argparse
import sys
from collections.abc import Sequence

from rlcardes.falconer.commands import cmd_evaluate, cmd_export, cmd_hunt, cmd_train


def _shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--cells", type=int, default=4000)
    parser.add_argument("--records", type=int, default=2950)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--episode", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--rollout", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=50)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlcardes.falconer")
    sub = parser.add_subparsers(dest="verb", required=True)

    train = sub.add_parser("train")
    _shared(train)
    train.add_argument("--out", default="runs/main")
    train.set_defaults(handler=cmd_train)

    hunt = sub.add_parser("hunt")
    _shared(hunt)
    hunt.set_defaults(handler=cmd_hunt)

    evaluate = sub.add_parser("evaluate")
    _shared(evaluate)
    evaluate.set_defaults(handler=cmd_evaluate)

    export = sub.add_parser("export")
    export.add_argument("--recipe", required=True)
    export.add_argument("--out", default="runs/persistnet.onnx")
    export.set_defaults(handler=cmd_export)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = args.handler
    return int(handler(args))


if __name__ == "__main__":
    sys.exit(main())
