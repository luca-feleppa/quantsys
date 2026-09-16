# Shared helpers for the forward collectors (01c IV, 01d L2, 01e trades) —
# extracted 2026-07-16 (boy-scout rule: third identical copy = extract).
# Single function: keyed-dedup parquet append with ATOMIC write
# (tmp + os.replace, THEORY.md §12.5 safety net): a mid-tick crash can never
# corrupt the accumulated history.
from pathlib import Path

import pandas as pd

from quantsys.utils.atomic_save import atomic_save_parquet


def append_parquet(path: Path, new_rows: pd.DataFrame, dedup_cols: list,
                   sort_col: str | None = None) -> int:
    # keyed-dedup append (keep='last': the new tick wins) + sort on
    # `sort_col` (default: first dedup key — historical 01c/01d behaviour;
    # 01e passes 'timestamp' since it dedups on trade_id).
    # Returns the file's total row count after the append.
    if new_rows.empty:
        return 0
    if path.exists():
        old = pd.read_parquet(path)
        merged = pd.concat([old, new_rows], ignore_index=True)
    else:
        merged = new_rows
    merged = merged.drop_duplicates(subset=dedup_cols, keep="last")
    merged = merged.sort_values(sort_col or dedup_cols[0]).reset_index(drop=True)
    atomic_save_parquet(merged, path, index=False)
    return len(merged)
