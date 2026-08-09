from __future__ import annotations

from typing import Any


try:
    from rich.console import Console
    from rich.table import Table
except ModuleNotFoundError:  # pragma: no cover - keeps non-training utilities usable before install.
    Console = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]


def get_console() -> Any:
    if Console is None:
        return None
    return Console()


def print_table(title: str, columns: list[str], rows: list[list[Any]]) -> None:
    console = get_console()
    if console is None or Table is None:
        print(title)
        print(" | ".join(columns))
        for row in rows:
            print(" | ".join(str(value) for value in row))
        return

    table = Table(title=title)
    for column in columns:
        table.add_column(column)
    for row in rows:
        table.add_row(*[str(value) for value in row])
    console.print(table)


def print_panel(message: str, title: str | None = None) -> None:
    console = get_console()
    if console is None:
        print(message)
        return

    from rich.panel import Panel

    console.print(Panel(message, title=title))
