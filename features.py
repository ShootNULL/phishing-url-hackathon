import math
import re
from collections import Counter
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import tldextract

_extract = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)  # без скачивания списка суффиксов

SUSPICIOUS_WORDS = [
    "login", "signin", "sign-in", "verify", "verification", "secure", "account", "update", "confirm",
    "bank", "banking", "password", "wallet", "webscr", "cmd", "billing", "support", "unlock", "suspend",
    "authenticate", "auth", "session", "invoice", "payment", "pay", "recover", "validation", "token",
    "office", "outlook", "microsoft", "apple", "icloud", "paypal", "amazon", "netflix", "facebook",
    "instagram", "whatsapp", "dropbox", "docusign", "chase", "wellsfargo", "itau", "bradesco",
    "santander", "bancolombia", "sbcglobal", "att", "yahoo", "gmail", "google", "adobe", "dhl", "fedex",
    "usps", "post", "telegram", "steam", "metamask", "binance", "coinbase", "wp-", "admin", ".php",
    "index.html", "free", "gift", "bonus", "airdrop",
]
SHORTENERS = {"bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "cutt.ly", "rb.gy", "shorturl.at",
              "tiny.cc", "rebrand.ly", "bit.do", "s.id", "qr.ae", "v.gd"}
HOSTING = {"web.app", "firebaseapp.com", "000webhostapp.com", "weebly.com", "wixsite.com", "duckdns.org",
           "github.io", "herokuapp.com", "glitch.me", "netlify.app", "vercel.app", "pages.dev", "workers.dev",
           "blogspot.com", "sites.google.com", "ngrok.io", "repl.co", "square.site", "webflow.io",
           "azurewebsites.net", "appspot.com", "r2.dev", "ipfs.io", "godaddysites.com", "myftp.org"}


def entropy(s):
    if not s:
        return 0.0
    c = Counter(s)
    n = len(s)
    return -sum(v / n * math.log2(v / n) for v in c.values())


def parse(url):
    u = url.strip()
    if "://" not in u:
        u = "http://" + u
    try:
        p = urlsplit(u)
        host = (p.hostname or "").lower()
        return p.scheme.lower(), host, p.path, p.query, p.fragment, p.port if _safe_port(p) else None
    except ValueError:
        return "", "", u, "", "", None


def _safe_port(p):
    try:
        return p.port is not None
    except ValueError:
        return False


def split_url(url):
    scheme, host, path, query, frag, port = parse(url)
    ext = _extract(host)
    reg = ext.top_domain_under_public_suffix if hasattr(ext, "top_domain_under_public_suffix") else ext.registered_domain
    return dict(scheme=scheme, host=host, path=path, query=query, frag=frag, port=port,
                sub=ext.subdomain, dom=ext.domain, suffix=ext.suffix, reg=reg or host)


def handcrafted(urls):
    rows = []
    for url in urls:
        s = split_url(url)
        host, path, query, sub, dom = s["host"], s["path"], s["query"], s["sub"], s["dom"]
        low = url.lower()
        f = {
            "len": len(url), "host_len": len(host), "path_len": len(path), "query_len": len(query),
            "frag_len": len(s["frag"]), "dom_len": len(dom), "sub_len": len(sub),
            "n_dots_host": host.count("."), "n_hyph_host": host.count("-"), "n_digit_host": sum(c.isdigit() for c in host),
            "n_sub_parts": len(sub.split(".")) if sub else 0,
            "n_slash": path.count("/"), "n_qparams": query.count("&") + (1 if query else 0),
            "n_digits": sum(c.isdigit() for c in url), "n_upper": sum(c.isupper() for c in url),
            "n_special": sum(not c.isalnum() for c in url),
            "n_at": url.count("@"), "n_tilde": url.count("~"), "n_pct": url.count("%"), "n_eq": url.count("="),
            "n_underscore": url.count("_"), "n_dslash": url[8:].count("//"),
            "https": int(s["scheme"] == "https"), "has_port": int(s["port"] is not None),
            "is_ip": int(bool(re.fullmatch(r"[\d.]+", host)) or ":" in host),
            "punycode": int("xn--" in host), "www": int(host.startswith("www.")),
            "trailing_slash": int(url.endswith("/")), "path_empty": int(path in ("", "/")),
            "ent_url": entropy(url), "ent_host": entropy(host), "ent_dom": entropy(dom),
            "digit_ratio_dom": sum(c.isdigit() for c in dom) / max(1, len(dom)),
            "vowel_ratio_dom": sum(c in "aeiou" for c in dom) / max(1, len(dom)),
            "shortener": int(s["reg"] in SHORTENERS), "hosting": int(s["reg"] in HOSTING or any(host.endswith("." + h) for h in HOSTING)),
            "n_susp": sum(w in low for w in SUSPICIOUS_WORDS),
            "susp_in_host": sum(w in host for w in SUSPICIOUS_WORDS),
            "http_in_path": int("http" in path.lower() or "www" in path.lower()),
            "com_in_sub": int(".com" in sub or "com-" in sub),
            "ext_php": int(path.lower().endswith(".php")), "ext_html": int(bool(re.search(r"\.html?$", path.lower()))),
            "max_tok_len": max((len(t) for t in re.split(r"[^A-Za-z0-9]", url) if t), default=0),
            "hex_long": int(bool(re.search(r"[0-9a-f]{20,}", low))),
        }
        rows.append(f)
    return pd.DataFrame(rows).astype(np.float32)
