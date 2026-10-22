# парсим html из *_with_content.csv -> cache/content_{train,test}.parquet
import csv
import os
import re
import sys
from multiprocessing import Pool
from urllib.parse import urlsplit

import pandas as pd
from selectolax.lexbor import LexborHTMLParser as HTMLParser

csv.field_size_limit(sys.maxsize)
SRC = "Архив 3"
MAX_HTML = 2_000_000
TEXT_CHARS = 3000

BRANDS = ["paypal", "microsoft", "office", "outlook", "apple", "icloud", "amazon", "netflix", "facebook", "instagram",
          "whatsapp", "google", "gmail", "yahoo", "dropbox", "docusign", "adobe", "dhl", "fedex", "usps", "chase",
          "wells fargo", "bank", "itau", "bradesco", "santander", "bancolombia", "att", "sbcglobal", "aol", "linkedin",
          "telegram", "steam", "metamask", "binance", "coinbase", "wallet", "sberbank", "сбербанк", "госуслуги"]
PHISH_WORDS = ["password", "passwd", "sign in", "signin", "log in", "login", "verify", "verification", "confirm",
               "account", "suspended", "unlock", "update your", "security", "billing", "credit card", "card number",
               "cvv", "ssn", "expir", "seed phrase", "recovery phrase", "пароль", "войти", "вход", "карт"]


def host_of(u):
    try:
        return (urlsplit(u).hostname or "").lower()
    except ValueError:
        return ""


def reg_of(h):
    parts = h.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else h


def features(args):
    url, html = args
    f = {"html_len": len(html), "html_empty": int(len(html.strip()) == 0)}
    html = html[:MAX_HTML]
    low = html.lower()
    page_reg = reg_of(host_of(url if "://" in url else "http://" + url))
    try:
        tree = HTMLParser(html)
    except Exception:
        tree = None
    title, text, actions, link_hosts = "", "", [], []
    if tree is not None and tree.body is not None or (tree is not None and tree.root is not None):
        t = tree.css_first("title")
        title = t.text(strip=True)[:300] if t else ""
        forms = tree.css("form")
        inputs = tree.css("input")
        f["n_forms"] = len(forms)
        f["n_inputs"] = len(inputs)
        f["n_pwd"] = sum((i.attributes.get("type") or "").lower() == "password" for i in inputs)
        f["n_hidden"] = sum((i.attributes.get("type") or "").lower() == "hidden" for i in inputs)
        f["n_email_in"] = sum((i.attributes.get("type") or "").lower() == "email" or "mail" in (i.attributes.get("name") or "").lower() for i in inputs)
        ext_act = 0
        for fm in forms:
            a = (fm.attributes.get("action") or "").strip()
            actions.append(a[:100])
            ah = host_of(a) if a.startswith("http") else ""
            ext_act += int(bool(ah) and reg_of(ah) != page_reg)
            f["act_php"] = f.get("act_php", 0) + int(".php" in a.lower())
            f["act_mailto"] = f.get("act_mailto", 0) + int(a.lower().startswith("mailto"))
            f["act_empty"] = f.get("act_empty", 0) + int(a in ("", "#", "about:blank"))
        f["ext_action"] = ext_act
        anchors = tree.css("a")
        n_ext = n_null = 0
        for a in anchors:
            h = (a.attributes.get("href") or "").strip()
            if h in ("", "#", "#!") or h.lower().startswith("javascript"):
                n_null += 1
            elif h.startswith("http"):
                hh = host_of(h)
                if reg_of(hh) != page_reg:
                    n_ext += 1
                    if len(link_hosts) < 40:
                        link_hosts.append(hh)
        f["n_a"] = len(anchors)
        f["a_ext_ratio"] = n_ext / max(1, len(anchors))
        f["a_null_ratio"] = n_null / max(1, len(anchors))
        f["n_script"] = len(tree.css("script"))
        f["n_ext_script"] = sum(1 for s in tree.css("script[src]") if (s.attributes.get("src") or "").startswith("http"))
        f["n_iframe"] = len(tree.css("iframe"))
        f["n_img"] = len(tree.css("img"))
        f["n_link_css"] = len(tree.css("link[rel=stylesheet]"))
        f["n_meta"] = len(tree.css("meta"))
        f["has_favicon"] = int(tree.css_first("link[rel*=icon]") is not None)
        f["meta_refresh"] = int(tree.css_first("meta[http-equiv=refresh]") is not None or tree.css_first("meta[http-equiv=Refresh]") is not None)
        for node in tree.css("script, style, noscript"):
            node.decompose()
        body = tree.body
        text = re.sub(r"\s+", " ", body.text(separator=" ") if body is not None else "").strip()
    f["text_len"] = len(text)
    f["title_len"] = len(title)
    tl = (title + " " + text[:5000]).lower()
    f["n_brand"] = sum(b in tl for b in BRANDS)
    f["n_phish_words"] = sum(w in tl for w in PHISH_WORDS)
    f["title_brand_not_in_url"] = int(any(b in title.lower() and b.replace(" ", "") not in url.lower() for b in BRANDS))
    f["js_atob"] = low.count("atob(") + low.count("unescape(") + low.count("eval(")
    f["js_location"] = low.count("window.location") + low.count("location.href") + low.count("location.replace")
    f["savepage"] = int("savepage" in low or "saved from url" in low or "httrack" in low)
    f["telegram_api"] = int("api.telegram.org" in low)
    f["parked"] = int(any(k in low for k in ("domain is for sale", "parked", "buy this domain", "coming soon", "adsense/domains")))
    f["err_page"] = int(any(k in low for k in ("404 not found", "403 forbidden", "account suspended", "suspendedpage", "index of /")))
    f["wp"] = int("wp-content" in low)
    f["copyright_year"] = max((int(y) for y in re.findall(r"(?:©|&copy;|copyright)\D{0,10}(20\d\d)", low)), default=0)
    f["ctext"] = " ".join([title, "|", " ".join(actions), "|", " ".join(link_hosts), "|", text[:TEXT_CHARS]])
    f["title"] = title
    return f


def rows(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        r = csv.reader(fh)
        header = next(r)
        ci = header.index("content")
        for row in r:
            yield row[0], row[ci] if len(row) > ci else ""


def chunks(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


if __name__ == "__main__":
    os.makedirs("cache", exist_ok=True)
    for split in ["test", "train"]:
        out = f"cache/content_{split}.parquet"
        if os.path.exists(out):
            continue
        urls, feats = [], []
        with Pool(8) as pool:
            for batch in chunks(rows(f"{SRC}/{split}_with_content.csv"), 2000):
                urls += [u for u, _ in batch]
                feats += pool.map(features, batch, chunksize=50)
                print(split, len(urls), flush=True)
        df = pd.DataFrame(feats).fillna(0)
        df.insert(0, "url", urls)
        df.to_parquet(out)
        print("wrote", out, df.shape)
