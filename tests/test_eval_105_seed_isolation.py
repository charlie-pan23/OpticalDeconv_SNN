from pathlib import Path

from eval.eval_105 import resolve_output_dir


def test_seed_indexed_output_isolated_from_legacy_directory(tmp_path: Path):
    indexed = resolve_output_dir(tmp_path, "cifar10dvs", 7, True)
    legacy = resolve_output_dir(tmp_path, "cifar10dvs", 7, False)
    assert indexed == tmp_path / "cifar10dvs" / "eval_105" / "seeds" / "seed_007"
    assert legacy == tmp_path / "cifar10dvs" / "eval_105"
    assert indexed != legacy
