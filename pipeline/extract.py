import os
import zipfile
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ===== 默认配置：以后要改路径，直接改这里 =====
_HERE = os.path.dirname(os.path.abspath(__file__))   # extract.py 所在目录（即项目根）

DEFAULT_DATA_ROOT = "/home/fund/data"
DEFAULT_A500_CSV  = os.path.join(_HERE, "..", "config", "a500.csv")
DEFAULT_OUTDIR    = os.path.join(_HERE, "..", "data")
DEFAULT_FORMAT    = "parquet"   # csv / parquet
DEFAULT_CHUNKSIZE = 200_000
DEFAULT_ENCODING  = "gb18030"
DONE_FILE_THRESHOLD = 500   # 某天目录里文件数 >= 500 认为已完成
SUMMARY_FILE_NAME = "_extract_day_summary.csv"
# ============================================


class TqdmReader:
    def __init__(self, f, pbar):
        self._f = f
        self._pbar = pbar

    def read(self, n=-1):
        b = self._f.read(n)
        if b:
            self._pbar.update(len(b))
        return b

    def readline(self, n=-1):
        b = self._f.readline(n)
        if b:
            self._pbar.update(len(b))
        return b

    def __iter__(self):
        for line in self._f:
            if line:
                self._pbar.update(len(line))
            yield line

    def close(self):
        return self._f.close()


def norm6(x) -> str:
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s = "".join(ch for ch in s if ch.isdigit())
    return s.zfill(6)


def read_zip_header_cols(zf, inner_csv, encoding="gb18030"):
    with zf.open(inner_csv) as f:
        header = f.readline().decode(encoding, errors="replace").rstrip("\n")
    cols = [c.strip() for c in header.split(",")]
    if cols and cols[-1] == "":
        cols = cols[:-1]
    return cols


def load_code_set(path_csv):
    df = pd.read_csv(path_csv)
    # 兼容 "code" 和 "SecurityID" 两种列名
    col = "SecurityID" if "SecurityID" in df.columns else "code"
    return set(df[col].astype(str).str.zfill(6).tolist())


def split_sh_sz(codes: set) -> tuple[set, set]:
    """按代码前缀拆分沪深：6开头为上交所，其余为深交所。"""
    sh = {c for c in codes if c.startswith("6")}
    sz = codes - sh
    return sh, sz


def load_code_sets_from_xls(path_xls):
    """
    从中证指数成分券权重 XLS 文件读取沪深股票代码集合。
    返回 (sh_codes, sz_codes)，代码均为 6 位补零字符串。
    """
    df = pd.read_excel(path_xls)
    code_col     = "成分券代码\nConstituent Code"
    exchange_col = "交易所\nExchange"
    codes = df[code_col].astype(str).str.strip().str.zfill(6)
    exchanges = df[exchange_col].astype(str).str.strip()
    sh_codes = set(codes[exchanges == "Shanghai"].tolist())
    sz_codes = set(codes[exchanges == "Shenzhen"].tolist())
    return sh_codes, sz_codes


def save_one_stock(df, out_path, fmt):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if fmt == "csv":
        df.to_csv(out_path, index=False)
    elif fmt == "parquet":
        df.to_parquet(out_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def list_2025_days(data_root):
    return list_days(data_root, prefix="2025")


def list_days(data_root, prefix=None):
    """列出 data_root 下所有8位数字日期目录，prefix 可过滤（如 '2025'、'202512'）。"""
    days = []
    for x in os.listdir(data_root):
        full = os.path.join(data_root, x)
        if len(x) == 8 and x.isdigit() and os.path.isdir(full):
            if prefix is None or x.startswith(prefix):
                days.append(x)
    return sorted(days)


def is_day_done(outdir, day, fmt, threshold=None):
    day_dir = os.path.join(os.path.expanduser(outdir), day)
    if not os.path.isdir(day_dir):
        return False
    ext = ".csv" if fmt == "csv" else ".parquet"
    cnt = len([f for f in os.listdir(day_dir) if f.endswith(ext)])
    limit = threshold if threshold is not None else DONE_FILE_THRESHOLD
    return cnt >= limit


def extract_market_to_stock_files(
    zpath,
    keep_cols,
    code_set,
    out_day_dir,
    day,
    fmt,
    encoding="gb18030",
    chunksize=200_000,
    show_progress=True,
):
    if not os.path.exists(zpath):
        raise FileNotFoundError(zpath)

    market_name = os.path.basename(zpath)
    market_tmp = defaultdict(list)

    with zipfile.ZipFile(zpath) as zf:
        inner_csv = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        cols = read_zip_header_cols(zf, inner_csv, encoding=encoding)
        n_cols = len(cols)

        missing = [c for c in keep_cols if c not in cols]
        if missing:
            raise ValueError(f"{market_name} missing columns: {missing}")

        total_bytes = zf.getinfo(inner_csv).file_size if (tqdm and show_progress) else None
        pbar = None
        if tqdm and show_progress:
            pbar = tqdm(
                total=total_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=market_name,
                leave=True,
            )

        with zf.open(inner_csv) as raw:
            wrapped = TqdmReader(raw, pbar) if pbar else raw
            reader = pd.read_csv(
                wrapped,
                chunksize=chunksize,
                encoding=encoding,
                encoding_errors="replace",
                index_col=False,
                skipinitialspace=True,
                usecols=list(range(n_cols)),
            )

            for ck in reader:
                ck["SecurityID"] = ck["SecurityID"].apply(norm6)
                ck = ck[ck["SecurityID"].isin(code_set)]
                if len(ck) == 0:
                    continue

                ck = ck[keep_cols].copy()
                ck.insert(0, "Date", day)

                for secid, g in ck.groupby("SecurityID", sort=False):
                    market_tmp[secid].append(g)

        if pbar:
            pbar.close()

    written_codes = set()
    ext = "csv" if fmt == "csv" else "parquet"
    for secid, parts in market_tmp.items():
        df = pd.concat(parts, ignore_index=True)
        out_path = os.path.join(out_day_dir, f"{secid}.{ext}")
        save_one_stock(df, out_path, fmt)
        written_codes.add(secid)

    return written_codes


def append_extract_day_summary(outdir, row_dict):
    summary_path = os.path.join(os.path.expanduser(outdir), SUMMARY_FILE_NAME)

    row_df = pd.DataFrame([row_dict])
    row_df["Date"] = row_df["Date"].astype(str)

    if os.path.exists(summary_path):
        old = pd.read_csv(summary_path, dtype={"Date": str})
        old["Date"] = old["Date"].astype(str)
        old = old[old["Date"] != str(row_dict["Date"])]
        new_df = pd.concat([old, row_df], ignore_index=True)
    else:
        new_df = row_df

    new_df["Date"] = new_df["Date"].astype(str)
    new_df = new_df.sort_values("Date").reset_index(drop=True)
    new_df.to_csv(summary_path, index=False)


def run_one_day(day, args, sh_codes, sz_codes, all_expected_codes):
    show_progress = not args.no_progress
    day_dir = os.path.join(args.data_root, day)
    if not os.path.isdir(day_dir):
        raise FileNotFoundError(day_dir)

    out_day_dir = os.path.join(os.path.expanduser(args.outdir), day)
    os.makedirs(out_day_dir, exist_ok=True)

    sh_file_keep = [
        "UpdateTime",
        "SecurityID",
        "PreCloPrice",
        "OpenPrice",
        "LastPrice",
        "Turnover",
        "LocalTime",
        "SeqNo",
        "TradVolume",
        "InstruStatus",
        "AskPrice1", "AskPrice2", "AskPrice3", "AskPrice4", "AskPrice5",
        "AskVolume1", "AskVolume2", "AskVolume3", "AskVolume4", "AskVolume5",
        "BidPrice1", "BidPrice2", "BidPrice3", "BidPrice4", "BidPrice5",
        "BidVolume1", "BidVolume2", "BidVolume3", "BidVolume4", "BidVolume5",
    ]

    sz_file_keep = [
        "UpdateTime",
        "SecurityID",
        "PreCloPrice",
        "OpenPrice",
        "LastPrice",
        "Turnover",
        "LocalTime",
        "SeqNo",
        "Volume",
        "TradingPhaseCode",
        "AskPrice1", "AskPrice2", "AskPrice3", "AskPrice4", "AskPrice5",
        "AskVolume1", "AskVolume2", "AskVolume3", "AskVolume4", "AskVolume5",
        "BidPrice1", "BidPrice2", "BidPrice3", "BidPrice4", "BidPrice5",
        "BidVolume1", "BidVolume2", "BidVolume3", "BidVolume4", "BidVolume5",
    ]

    sh_zip = os.path.join(day_dir, "mdl_4_4_0.csv.zip")
    sz_zip = os.path.join(day_dir, "mdl_6_28_0.csv.zip")

    print("Extracting SH...")
    sh_written_codes = extract_market_to_stock_files(
        sh_zip, sh_file_keep, sh_codes, out_day_dir, day,
        fmt=args.format,
        encoding=args.encoding,
        chunksize=args.chunksize,
        show_progress=show_progress,
    )
    print(f"SH written stock files: {len(sh_written_codes)}")

    print("Extracting SZ...")
    sz_written_codes = extract_market_to_stock_files(
        sz_zip, sz_file_keep, sz_codes, out_day_dir, day,
        fmt=args.format,
        encoding=args.encoding,
        chunksize=args.chunksize,
        show_progress=show_progress,
    )
    print(f"SZ written stock files: {len(sz_written_codes)}")

    extracted_codes = sh_written_codes | sz_written_codes
    extracted_count = len(extracted_codes)

    missing_codes = sorted(all_expected_codes - extracted_codes)
    missing_count = len(missing_codes)

    status = "OK" if extracted_count == len(all_expected_codes) else "INCOMPLETE"

    print(f"Done. Output dir: {out_day_dir}. Total files: {extracted_count}")
    print(f"Expected: {len(all_expected_codes)}, Extracted: {extracted_count}, Missing: {missing_count}")

    return {
        "Date": day,
        "ExpectedStocks": len(all_expected_codes),
        "ExtractedStocks": extracted_count,
        "MissingCount": missing_count,
        "MissingStocks": ";".join(missing_codes),
        "Status": status,
    }


def _day_worker(pack):
    """ProcessPoolExecutor worker：处理单天，返回 summary dict。并行时静默 stdout。"""
    import io, sys
    day, args, sh_codes, sz_codes, all_expected_codes = pack
    sys.stdout = io.StringIO()   # 静默 worker 内部所有 print
    try:
        result = run_one_day(day, args, sh_codes, sz_codes, all_expected_codes)
        sys.stdout = sys.__stdout__
        return result
    except Exception as e:
        sys.stdout = sys.__stdout__
        print(f"FAILED {day}: {e}")
        return {
            "Date": day,
            "ExpectedStocks": len(all_expected_codes),
            "ExtractedStocks": 0,
            "MissingCount": len(all_expected_codes),
            "MissingStocks": ";".join(sorted(all_expected_codes)),
            "Status": f"FAIL: {str(e)}",
        }


def main():
    ap = argparse.ArgumentParser(description="Extract A500 raw snapshots, one file per stock")
    ap.add_argument("--day", default=None, help="Single day, e.g. 20250603")
    ap.add_argument("--days", nargs="*", default=None, help="Explicit multiple days")
    ap.add_argument("--start-day", default=None, help="Only run days >= this day, e.g. 20250123")
    ap.add_argument("--n", type=int, default=None, help="Run first N trading days in 2025")
    ap.add_argument("--force", action="store_true", help="Re-run even if day output already exists")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--universe-csv", default=DEFAULT_A500_CSV,
                    help="股票池 CSV（含 SecurityID 列，6位补零）；默认 config/a500.csv")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--format", choices=["csv", "parquet"], default=DEFAULT_FORMAT)
    ap.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    ap.add_argument("--encoding", default=DEFAULT_ENCODING)
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help="并行进程数（按天并行，默认 1）")
    args = ap.parse_args()

    all_expected_codes = load_code_set(args.universe_csv)
    sh_codes, sz_codes = split_sh_sz(all_expected_codes)
    print(f"股票池来自 {args.universe_csv}：共 {len(all_expected_codes)} 只（SH={len(sh_codes)}, SZ={len(sz_codes)}）")

    # 选日期
    if args.day:
        selected_days = [args.day]
    elif args.days:
        selected_days = args.days
    else:
        all_2025 = list_2025_days(args.data_root)
        selected_days = all_2025 if args.n is None else all_2025[:args.n]

    # 过滤不存在日期
    all_dirs = set(os.listdir(args.data_root))
    selected_days = [d for d in selected_days if d in all_dirs]

    if args.start_day:
        selected_days = [d for d in selected_days if d >= args.start_day]
        
    all_2025 = list_2025_days(args.data_root)

    done_days = []
    todo_days = []
    for d in selected_days:
        if (not args.force) and is_day_done(args.outdir, d, args.format):
            done_days.append(d)
        else:
            todo_days.append(d)

    print("=" * 60)
    print(f"2025 trading days total : {len(all_2025)}")
    print(f"Selected this run       : {len(selected_days)}")
    print(f"Already done            : {len(done_days)}")
    print(f"Need to run now         : {len(todo_days)}")
    if done_days:
        print("Done days:")
        print("  " + " ".join(done_days))
    if todo_days:
        print("To run days:")
        print("  " + " ".join(todo_days))
    print("=" * 60)

    if not todo_days:
        print("Nothing to do.")
        return

    ok_days = []
    fail_days = []
    all_summaries = []

    packs = [
        (day, args, sh_codes, sz_codes, all_expected_codes)
        for day in todo_days
    ]

    if args.workers == 1:
        for i, pack in enumerate(packs, 1):
            day = pack[0]
            print(f"\n[{i}/{len(todo_days)}] === processing {day} ===")
            try:
                summary = _day_worker(pack)
                all_summaries.append(summary)
                if summary["Status"].startswith("FAIL"):
                    fail_days.append(day)
                else:
                    ok_days.append(day)
            except KeyboardInterrupt:
                print("\nInterrupted by user.")
                break
    else:
        print(f"并行模式：workers={args.workers}")
        # 并行时关掉每只股票的 per-stock 进度条，只显示整体天进度
        import copy
        args_quiet = copy.copy(args)
        args_quiet.no_progress = True
        packs = [(day, args_quiet, sh_codes, sz_codes, all_expected_codes) for day in todo_days]
        pbar = tqdm(total=len(todo_days), desc="extract days", unit="day") if tqdm else None
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                future_to_day = {pool.submit(_day_worker, p): p[0] for p in packs}
                for fut in as_completed(future_to_day):
                    day = future_to_day[fut]
                    try:
                        summary = fut.result()
                        all_summaries.append(summary)
                        if summary["Status"].startswith("FAIL"):
                            fail_days.append(day)
                        else:
                            ok_days.append(day)
                    except Exception as e:
                        print(f"FAILED {day}: {e}")
                        fail_days.append(day)
                    if pbar:
                        pbar.update(1)
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            if pbar:
                pbar.close()

    for summary in all_summaries:
        append_extract_day_summary(args.outdir, summary)

    print("\n" + "=" * 60)
    print(f"Finished. OK={len(ok_days)}, FAIL={len(fail_days)}")
    if ok_days:
        print("OK days:")
        print("  " + " ".join(sorted(ok_days)))
    if fail_days:
        print("FAIL days:")
        print("  " + " ".join(sorted(fail_days)))
    print("=" * 60)


if __name__ == "__main__":
    main()