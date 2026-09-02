"""`freedom` command line. Every command is thin: parse args, call a module, print a table."""

from __future__ import annotations

import math

import typer
from rich.console import Console
from rich.table import Table

from .config import get_settings

app = typer.Typer(help="Post-earnings price-action harness for Hyperliquid equity perpetuals.")
console = Console()


def _csv(arg: str) -> list[str]:
    return [x.strip() for x in arg.split(",") if x.strip()]


def _blank(v: object) -> bool:
    import pandas as pd

    return v is None or v is pd.NaT or v is pd.NA or (isinstance(v, float) and math.isnan(v))


def _print_frame(df, title: str) -> None:
    """A DataFrame as a rich table: timestamps to the minute, missing values blank."""
    import pandas as pd

    table = Table(title=title)
    for col in df.columns:
        table.add_column(str(col))
    for row in df.itertuples(index=False):
        table.add_row(*("" if _blank(v) else v.strftime("%Y-%m-%d %H:%M")
                        if isinstance(v, pd.Timestamp) else str(v) for v in row))
    console.print(table)


@app.callback()
def _root() -> None:
    pass


@app.command()
def universe(verify_only: bool = typer.Option(False, help="Print only rows needing human verification")) -> None:
    """Pull Hyperliquid markets, classify them, write data/universe.parquet."""
    raise NotImplementedError


@app.command()
def archive(
    intervals: str = typer.Option("1m,5m,15m,1h", help="Comma-separated candle intervals"),
    markets: str = typer.Option("", help="Comma-separated markets; default: every market in data/universe.parquet"),
) -> None:
    """Archive recent candles and funding for every universe market (run at least every 3 days)."""
    import pandas as pd

    from .data import archive as archive_mod
    from .schemas import U

    s = get_settings()
    if markets:
        names = _csv(markets)
    elif s.universe_path.exists():
        names = pd.read_parquet(s.universe_path)[U.market].dropna().unique().tolist()
    else:
        console.print(f"{s.universe_path} not found: run `freedom universe` first or pass --markets",
                      style="red", markup=False)
        raise typer.Exit(code=2)
    summary = archive_mod.archive_markets(s, names, _csv(intervals))
    _print_frame(summary, title=f"freedom archive: {len(names)} markets -> {s.archive_dir}")
    n_problems = int(summary["error"].notna().sum())
    if n_problems:
        console.print(f"{n_problems} of {len(summary)} items reported an error or a coverage gap "
                      "(see the error column).", style="yellow")


@app.command()
def events(since: str = typer.Option("2022-01-01"), underlyings: str = typer.Option("", help="Comma-separated subset")) -> None:
    """Build the earnings event table with resolved release times."""
    raise NotImplementedError


@app.command()
def dataset(decision_times: str = typer.Option("pre_5m,post_15m,post_30m")) -> None:
    """Compute targets and features; write data/dataset.parquet."""
    raise NotImplementedError


@app.command()
def evaluate(models: str = typer.Option("zero,historical_mean,sign_of_reaction,ridge,lightgbm"),
             decision_times: str = typer.Option("pre_5m,post_30m")) -> None:
    """Walk-forward evaluation with cost-aware simulation; writes reports/<run_id>/."""
    raise NotImplementedError


@app.command()
def optimize(n_trials: int = typer.Option(50), objective: str = typer.Option("brier")) -> None:
    """Optuna search over models, features and decision time; honest holdout score at the end."""
    raise NotImplementedError


@app.command()
def train(model: str = typer.Option("lightgbm"), decision_time: str = typer.Option("post_30m")) -> None:
    """Fit a model on all events and save it under data/models/."""
    raise NotImplementedError


@app.command()
def predict(market: str, decision_time: str = typer.Option("post_30m")) -> None:
    """Live prediction for a market's most recent/next event using data as of now."""
    raise NotImplementedError


@app.command()
def upcoming(days: int = typer.Option(14)) -> None:
    """List upcoming earnings events in the event universe."""
    raise NotImplementedError


@app.command()
def status() -> None:
    """Show configured keys, budgets used today, archive coverage and dataset sizes."""
    s = get_settings()
    console.print({"data_dir": str(s.data_dir), "fmp_key": bool(s.fmp_api_key),
                   "alphavantage_key": bool(s.alphavantage_api_key), "anthropic_key": bool(s.anthropic_api_key)})


if __name__ == "__main__":
    app()
