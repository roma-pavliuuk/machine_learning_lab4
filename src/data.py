import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "config_unsupervised.yaml"


def load_config(path=CONFIG) -> Dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def reference_id(cfg: Dict) -> Dict[str, Optional[str]]:
    ref = cfg["reference"]
    pointer = (ROOT / ref["repo"] / f"{ref['path']}.dvc").resolve()
    md5 = yaml.safe_load(pointer.read_text(encoding="utf-8"))["outs"][0]["md5"]
    if md5 != ref["expected_md5"]:
        raise ValueError(f"Еталон змінився: md5 {md5} ≠ очікуваного {ref['expected_md5']}")
    return {"rev": ref["rev"], "md5": md5, "pointer": str(pointer)}


def load_reference(cfg: Dict) -> Tuple[pd.DataFrame, pd.Series]:
    import dvc.api

    ref = cfg["reference"]
    repo = str((ROOT / ref["repo"]).resolve())
    with dvc.api.open(ref["path"], repo=repo, rev=ref["rev"]) as f:
        content = f.read()
    actual = hashlib.md5(content.encode("utf-8")).hexdigest()
    if actual != ref["expected_md5"]:
        raise ValueError(f"Прочитаний еталон має md5 {actual}, очікувано {ref['expected_md5']}")
    from io import StringIO

    df = pd.read_csv(StringIO(content))
    y_sealed = df.pop(cfg["target"])
    return df, y_sealed
