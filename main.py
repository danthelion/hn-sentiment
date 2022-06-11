import dask.dataframe as dd
import numpy as np
import pandas as pd
import pandas_gbq
from google.oauth2 import service_account
from matplotlib import pyplot as plt
from transformers import pipeline


def chunk(s):
    return s.value_counts()


def agg(s):
    return s.apply(lambda s: s.groupby(level=-1).sum())


def finalize(s):
    level = list(range(s.index.nlevels - 1))
    return s.groupby(level=level).apply(
        lambda s: s.reset_index(level=level, drop=True).idxmax()
    )


max_occurrence_agg = dd.Aggregation("mode", chunk, agg, finalize)


def get_comments():
    credentials = service_account.Credentials.from_service_account_file(
        "client-secret.json",
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    query = """
    SELECT
      id AS id,
      text AS comments,
      DATETIME(timestamp) AS datetime
    FROM
      `bigquery-public-data.hacker_news.full`
    WHERE
      DATE(timestamp) > DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH)
    ORDER BY datetime ASC;
    """
    df = pandas_gbq.read_gbq(
        query, project_id="hn-sentiment-352818", credentials=credentials, index_col="id"
    )

    df.to_csv("comments_1M.csv")

    # Preprocess
    df = df.dropna()
    df["comments"] = df["comments"].str.slice(0, 512)

    # Sample 100 comments per hour so we have a fairly balanced dataset
    df = df.groupby(pd.to_datetime(df.datetime).dt.hour, group_keys=False).apply(
        lambda x: x.sample(1000)
    )

    df = df[df["comments"].str.contains("(?i)meta")]
    return df


if __name__ == "__main__":
    sentiment_task = pipeline("sentiment-analysis")

    df = get_comments()

    ddf = dd.from_pandas(df, npartitions=50)
    ddf["sentiment"] = ddf.apply(
        lambda row: sentiment_task(row["comments"])[0]["label"],
        meta=(None, "object"),
        axis=1,
    ).compute()

    # Aggregate by mode
    ddf["datetime"] = ddf["datetime"].astype("datetime64[D]")
    res = (
        ddf[["sentiment", "datetime"]]
        .groupby("datetime", group_keys=False)
        .agg(max_occurrence_agg)
        .compute()
    )

    # Read meta stock
    df_meta = pd.read_csv("META.csv")
    df_meta["Date"] = pd.to_datetime(df_meta["Date"])

    df_meta = df_meta.join(res, on="Date", how="left")

    # Select first week from the dataframe
    df_meta = df_meta[df_meta["Date"] <= pd.to_datetime("2022-05-18")]

    x = np.arange(0, len(df_meta))
    fig, ax = plt.subplots(1, figsize=(12, 6))
    for idx, val in df_meta.iterrows():
        color = "#2CA453"
        if val["Open"] > val["Close"]:
            color = "#F04730"

        plt.plot([x[idx], x[idx]], [val["Low"], val["High"]], color=color)
        # Annotate sentiment on the graph
        ax.annotate(val["sentiment"], (x[idx], val["High"]), color=color, fontsize=20)
        plt.plot([x[idx], x[idx] - 0.1], [val["Open"], val["Open"]], color=color)
        plt.plot([x[idx], x[idx] + 0.1], [val["Close"], val["Close"]], color=color)

    # ticks
    plt.xticks(x, df_meta.Date.dt.date)
    ax.set_xticks(x, minor=True, rotation=45)

    # labels
    plt.ylabel("USD")
    # grid
    ax.xaxis.grid(color="black", linestyle="dashed", which="both", alpha=0.1)
    # remove spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # title
    plt.title("$META x Hackernews comment sentiment (+ / -)", loc="center", fontsize=20)

    plt.savefig("meta_stock.png")

    plt.show()
