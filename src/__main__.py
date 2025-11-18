"""Entry point for running the project as a module."""

from . import data_prep, evaluation, features, models, plotting


def main() -> None:
    """Run a lightweight import check to verify the project skeleton."""
    imported_modules = [
        data_prep.__name__,
        features.__name__,
        models.__name__,
        evaluation.__name__,
        plotting.__name__,
    ]
    print("ML NFL drift project package is ready. Imported modules:")
    for name in imported_modules:
        print(f" - {name}")


if __name__ == "__main__":
    main()
