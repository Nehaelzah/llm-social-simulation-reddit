"""
collect_reddit_streaming_price_ranges.py

Collect Reddit posts and comments related to streaming-service price hikes
using Reddit public JSON endpoints (no OAuth, no PRAW).

Updated for capstone:
- Netflix: collect from 2024-05-01 to now
- Disney Plus: collect from 2024-03-01 to now
- stricter relevance filtering to reduce noisy threads
"""

import requests
import time
import json
import datetime
import os
import csv

HEADERS = {
    "User-Agent": "academic-research-bot/0.2 (UQ student capstone, non-commercial research)"
}

REQUEST_DELAY = 1.2
MIN_TEXT_LEN = 10

NOW_UTC = datetime.datetime.now(datetime.timezone.utc)

COLLECTION_TARGETS = [
    {
        "target_id": "NETFLIX_PRICE_RANGE",
        "platform_name": "Netflix",
        "start_time": datetime.datetime(2024, 5, 1, 0, 0, 0, tzinfo=datetime.timezone.utc),
        "end_time": NOW_UTC,
        "platform_terms": ["netflix"],
        "price_terms": [
            "price", "pricing", "price increase", "price hike", "subscription",
            "subscription cost", "cost", "monthly price", "fee", "plan",
            "more expensive", "expensive", "ad tier", "ads", "cancel", "cancelling"
        ],
        "queries": [
            "Netflix price increase",
            "Netflix price hike",
            "Netflix subscription cost",
            "Netflix monthly price",
            "Netflix more expensive",
            "Netflix plan price",
            "Netflix ad tier price",
            "Netflix cancel because price"
        ],
        "subreddits": ["netflix", "cordcutters", "streaming"]
    },
    {
        "target_id": "DISNEYPLUS_PRICE_RANGE",
        "platform_name": "DisneyPlus",
        "start_time": datetime.datetime(2024, 3, 1, 0, 0, 0, tzinfo=datetime.timezone.utc),
        "end_time": NOW_UTC,
        "platform_terms": ["disney plus", "disney+", "disneyplus"],
        "price_terms": [
            "price", "pricing", "price increase", "price hike", "subscription",
            "subscription cost", "cost", "monthly price", "fee", "plan",
            "more expensive", "expensive", "bundle", "ad tier", "ads", "cancel", "cancelling"
        ],
        "queries": [
            "Disney Plus price increase",
            "Disney+ price hike",
            "Disney Plus subscription cost",
            "Disney+ monthly price",
            "Disney Plus more expensive",
            "Disney Plus bundle price",
            "Disney+ ad tier price",
            "Disney Plus cancel because price"
        ],
        "subreddits": ["DisneyPlus", "cordcutters", "streaming"]
    }
]


def within_window(ts, start_time, end_time):
    return start_time <= ts <= end_time


def reddit_get(url, params=None, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=20)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 60))
                print(f"    Rate limited. Waiting {wait}s")
                time.sleep(wait)
                continue
            return r
        except requests.RequestException as e:
            print(f"    Request error ({attempt + 1}/{retries}): {e}")
            time.sleep(5)
    return None


def text_has_any(text, terms):
    text = (text or "").lower()
    return any(term.lower() in text for term in terms)


def is_relevant_post(title, selftext, target):
    """
    Require:
    - at least one platform-related term
    - at least one price/cost-related term
    """
    combined = f"{title} {selftext}".lower()

    has_platform = text_has_any(combined, target["platform_terms"])
    has_price = text_has_any(combined, target["price_terms"])

    return has_platform and has_price


def search_posts(subreddit, query, target):
    """
    Search subreddit posts using public search.json endpoint.
    Results are filtered locally into the target date window and relevance rules.
    """
    start_time = target["start_time"]
    end_time = target["end_time"]

    posts = []
    seen = set()

    for timefilter in ["year", "all"]:
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
        params = {
            "q": query,
            "sort": "new",
            "restrict_sr": "on",
            "limit": 100,
            "t": timefilter,
        }

        r = reddit_get(url, params=params)
        time.sleep(REQUEST_DELAY)

        if not r or r.status_code != 200:
            print(f"    Search failed for r/{subreddit} | {query} | status={r.status_code if r else 'no response'}")
            continue

        try:
            data = r.json()
        except Exception as e:
            print(f"    JSON parse error: {e}")
            continue

        children = data.get("data", {}).get("children", [])

        for child in children:
            p = child.get("data", {})
            post_id = p.get("id", "")
            if not post_id or post_id in seen:
                continue

            created_utc = p.get("created_utc", 0)
            post_ts = datetime.datetime.fromtimestamp(created_utc, tz=datetime.timezone.utc)

            title = p.get("title", "")
            selftext = p.get("selftext", "")

            if within_window(post_ts, start_time, end_time) and is_relevant_post(title, selftext, target):
                posts.append({
                    "id": post_id,
                    "title": title,
                    "selftext": selftext,
                    "score": p.get("score", 0),
                    "num_comments": p.get("num_comments", 0),
                    "created_utc": created_utc,
                    "ts": post_ts,
                    "subreddit": subreddit,
                    "permalink": p.get("permalink", ""),
                    "url": p.get("url", ""),
                    "query_used": query
                })
                seen.add(post_id)

    return posts


def collect_comments(post, target):
    """
    Collect comments from a single Reddit thread using comments/{post_id}.json
    """
    post_id = post["id"]
    subreddit = post["subreddit"]
    start_time = target["start_time"]
    end_time = target["end_time"]

    url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
    params = {"limit": 500, "depth": 10}

    r = reddit_get(url, params=params)
    time.sleep(REQUEST_DELAY)

    if not r or r.status_code != 200:
        return []

    try:
        data = r.json()
    except Exception:
        return []

    rows = []

    # keep original post as one row
    post_text = (post["title"] + " " + post["selftext"]).strip()
    if len(post_text) >= MIN_TEXT_LEN and within_window(post["ts"], start_time, end_time):
        rows.append({
            "target_id": target["target_id"],
            "platform_name": target["platform_name"],
            "start_time": target["start_time"].isoformat(),
            "end_time": target["end_time"].isoformat(),
            "subreddit": subreddit,
            "thread_id": post_id,
            "thread_title": post["title"],
            "comment_id": post_id,
            "parent_id": None,
            "timestamp": post["ts"].isoformat(),
            "text": post_text,
            "score": post["score"],
            "depth": 0,
            "source_type": "post",
            "num_comments": post["num_comments"],
            "query_used": post["query_used"],
            "thread_permalink": f"https://www.reddit.com{post['permalink']}" if post["permalink"] else "",
        })

    def walk(comments, depth=1):
        for item in comments:
            if item.get("kind") != "t1":
                continue

            c = item.get("data", {})
            c_id = c.get("id", "")
            c_ts_raw = c.get("created_utc", 0)
            c_text = c.get("body", "")

            if not c_ts_raw:
                continue

            c_ts = datetime.datetime.fromtimestamp(c_ts_raw, tz=datetime.timezone.utc)

            if (
                len(c_text) >= MIN_TEXT_LEN
                and c_text not in ("[deleted]", "[removed]")
                and within_window(c_ts, start_time, end_time)
            ):
                rows.append({
                    "target_id": target["target_id"],
                    "platform_name": target["platform_name"],
                    "start_time": target["start_time"].isoformat(),
                    "end_time": target["end_time"].isoformat(),
                    "subreddit": subreddit,
                    "thread_id": post_id,
                    "thread_title": post["title"],
                    "comment_id": c_id,
                    "parent_id": c.get("parent_id", ""),
                    "timestamp": c_ts.isoformat(),
                    "text": c_text,
                    "score": c.get("score", 0),
                    "depth": depth,
                    "source_type": "comment",
                    "num_comments": post["num_comments"],
                    "query_used": post["query_used"],
                    "thread_permalink": f"https://www.reddit.com{post['permalink']}" if post["permalink"] else "",
                })

            replies = c.get("replies", {})
            if isinstance(replies, dict):
                walk(replies.get("data", {}).get("children", []), depth + 1)

    if len(data) >= 2:
        comment_listing = data[1].get("data", {}).get("children", [])
        walk(comment_listing, depth=1)

    return rows


def save_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    all_rows = []
    collection_log = []

    for target in COLLECTION_TARGETS:
        print(f"\n=== Collecting {target['target_id']} ===")
        print(f"Platform: {target['platform_name']}")
        print(f"Start: {target['start_time'].isoformat()}")
        print(f"End:   {target['end_time'].isoformat()}")

        target_rows = []
        post_ids_seen = set()
        total_search_calls = 0
        total_comment_calls = 0
        kept_posts = []

        for sub in target["subreddits"]:
            sub_posts = []

            for query in target["queries"]:
                print(f'  Searching r/{sub} for "{query}" ...')
                posts = search_posts(sub, query, target)
                total_search_calls += 2  # year + all

                new_posts = [p for p in posts if p["id"] not in post_ids_seen]
                post_ids_seen.update(p["id"] for p in new_posts)
                sub_posts.extend(new_posts)

                print(f"    Found {len(new_posts)} relevant new posts in date range")

            kept_posts.extend(sub_posts)

            print(f"  Collecting comments from {len(sub_posts)} kept posts in r/{sub} ...")
            for post in sub_posts:
                print(f"    Post {post['id']} | {post['title'][:90]}")
                rows = collect_comments(post, target)
                total_comment_calls += 1
                target_rows.extend(rows)
                print(f"      Collected {len(rows)} rows")

        fieldnames = [
            "target_id",
            "platform_name",
            "start_time",
            "end_time",
            "subreddit",
            "thread_id",
            "thread_title",
            "comment_id",
            "parent_id",
            "timestamp",
            "text",
            "score",
            "depth",
            "source_type",
            "num_comments",
            "query_used",
            "thread_permalink",
        ]

        if target_rows:
            raw_path = f"data/raw/reddit_{target['target_id']}.csv"
            save_csv(raw_path, target_rows, fieldnames)
            print(f"Saved target file: {raw_path}")

        all_rows.extend(target_rows)

        collection_log.append({
            "target_id": target["target_id"],
            "platform_name": target["platform_name"],
            "start_time": target["start_time"].isoformat(),
            "end_time": target["end_time"].isoformat(),
            "n_rows": len(target_rows),
            "n_unique_threads": len(set(r["thread_id"] for r in target_rows)) if target_rows else 0,
            "search_calls_estimated": total_search_calls,
            "comment_calls_estimated": total_comment_calls,
            "kept_thread_titles": list(sorted(set(p["title"] for p in kept_posts)))
        })

    if all_rows:
        combined_fieldnames = [
            "target_id",
            "platform_name",
            "start_time",
            "end_time",
            "subreddit",
            "thread_id",
            "thread_title",
            "comment_id",
            "parent_id",
            "timestamp",
            "text",
            "score",
            "depth",
            "source_type",
            "num_comments",
            "query_used",
            "thread_permalink",
        ]

        combined_path = "data/processed/reddit_streaming_price_ranges_combined.csv"
        save_csv(combined_path, all_rows, combined_fieldnames)
        print(f"\nSaved combined dataset: {combined_path}")

    log_path = "data/processed/collection_log_price_ranges.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(collection_log, f, indent=2)

    print(f"Saved collection log: {log_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()