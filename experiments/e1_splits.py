"""Create edition-grouped train/val/test splits (temporal + random) -> data/splits.json.

Run:  python experiments/e1_splits.py
"""
from _common import config  # noqa: F401  (sets sys.path)

from src import corpus, splits


def main():
    df = corpus.load_pages()
    s = splits.make_and_save(df)
    for scheme, parts in s.items():
        df[scheme] = splits.assign(df, scheme)
        print(f"\n{scheme}:")
        print(df.groupby([scheme, "paper"]).agg(editions=("edition", "nunique"), pages=("page_id", "size"),
                                                  years=("year", lambda y: f"{y.min()}-{y.max()}")))


if __name__ == "__main__":
    main()
