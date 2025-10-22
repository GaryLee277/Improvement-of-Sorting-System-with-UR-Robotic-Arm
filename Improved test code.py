# -*- coding: utf-8 -*-
"""
test.py — Batch OCR & Pairing for Insoles (single-file version)
v9.0-keywords+autocorrect+pairlog

Improvements:
  - Keyword bag to aid pairing: order-invariant, tolerant (soft Jaccard)
  - Date auto-correction: 67/25 -> 07/25; 97125 -> 07/25; and such 5-digit numbers are NOT treated as codes
  - 6-digit code multi-pass voting + ±90° rotation channels
  - Clinic collection "collect first, clean later": allow digits and dots, then standardize by stripping numbers/dates
  - Name cleaning: allow '.' and '-'; remove common logo noise (EVA/VA)
  - Logging: each line includes OCR tokens; pairing attempts add a separate line with similarities/scores; export pairs_log.csv
"""

import os, sys, json, time, re, base64, hashlib
from difflib import SequenceMatcher
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional
import requests
import numpy as np
import cv2
import csv
from datetime import datetime

# =========================
# Quick-edit test path
# =========================
TEST_IMAGE_DIR = r"Add an image directory"

# =========================
# Config (defaults; can be overridden by CLI)
# =========================

# IMPORTANT: do NOT hardcode your API key here. Set env var GOOGLE_API_KEY before running.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")

MAX_PAIR_NUMBER   = 70
PENDING_IDS       = (68, 69)
PENDING_START     = 68
OCR_TIMEOUT       = 25
SAVE_DIR          = "test_outputs"
CACHE_DIR         = "ocr_cache"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

USE_CACHE             = True
POSTPASS_MODE         = "soft"   # off / soft / strict
POSTPASS_MIN_SCORE    = 0.84
LIGHT_ROI_MAX         = 6
HEAVY_ROI_MAX         = 8

# Thresholds (CLI-overridable)
NAME_STRICT_SIM       = 0.94   # strong name-candidate threshold (used when at least one side has no code)
NAME_PAIR_SIM         = 0.92   # hard check: name similarity for same-slot
CODE_MAX_HAMMING      = 2      # max allowed digit difference when both sides have codes

# Keyword-bag similarity thresholds
TOKEN_SIM_THR         = 0.82   # per-token tolerance threshold
KEYWORDS_PAIR_SIM     = 0.80   # hard check: keyword similarity to allow same-slot

# Review band (samples in this band go to CSV for manual review)
REVIEW_BAND_LOW       = 0.85
REVIEW_BAND_HIGH      = 0.90

# Feature toggles (can be disabled by CLI)
ENABLE_VOTE_PREPROC   = True   # deskew + multi-binarization pixel voting
ENABLE_OCR_ENSEMBLE   = True   # multi-engine (Google + Tesseract) voting

# Internal toggles
_PENDING_TOGGLE = 0

# Blocking index
BLOCKS: Dict[str, set] = {}    # key -> set(slot_id)

# Review cache
REVIEW_ROWS: List[Tuple[int, Optional[str], str, Optional[str], str, float]] = []

# Pairing logs
PAIR_LOGS: List[Dict[str, object]] = []

# =========================
# Utils
# =========================

def imread_unicode(path: str, flags=cv2.IMREAD_COLOR):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return cv2.imread(path, flags)

def imencode_ext(ext: str, img) -> bytes:
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError("imencode failed")
    return buf.tobytes()

def fmt_kw_list(kws: List[str]) -> str:
    """Pretty-print as OCR tokens: ['A','B',...]"""
    if not kws: return "[]"
    inner = ", ".join([f"'{k}'" for k in kws])
    return f"[{inner}]"

# =========================
# OCR (Google + cache)
# =========================

def _cache_key(img_bytes: bytes, pass_tag: str) -> str:
    h = hashlib.md5()
    h.update(img_bytes); h.update(pass_tag.encode("utf-8"))
    return h.hexdigest()

def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.json")

def vision_ocr_image_bytes(img_bytes: bytes, pass_tag: str) -> dict:
    key = _cache_key(img_bytes, pass_tag)
    path = _cache_path(key)
    if USE_CACHE and os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    payload = {
        "requests": [{
            "image": {"content": base64.b64encode(img_bytes).decode("UTF-8")},
            "features": [{"type": "TEXT_DETECTION"}]
        }]}
    url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_API_KEY}"
    resp = requests.post(url, json=payload, timeout=OCR_TIMEOUT)
    resp.raise_for_status()
    js = resp.json()

    if USE_CACHE:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(js, f, ensure_ascii=False)
        except Exception:
            pass

    return js

def extract_text_tokens(ocr_response: dict) -> List[str]:
    try:
        full = ocr_response["responses"][0]["textAnnotations"][0]["description"]
        tokens = re.split(r"\s+", full.strip())
        return tokens
    except Exception:
        return []

def extract_full_text(ocr_response: dict) -> str:
    try:
        return ocr_response["responses"][0]["textAnnotations"][0]["description"]
    except Exception:
        return ""

def ocr_tokens_from_img(img, pass_tag: str) -> Tuple[List[str], str]:
    bjpg = imencode_ext(".jpg", img)
    ocr = vision_ocr_image_bytes(bjpg, pass_tag)
    return extract_text_tokens(ocr), extract_full_text(ocr)

# =========================
# Imaging preprocess (deskew + multi-binarization voting)
# =========================

def deskew(gray):
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 80)
    if lines is None:
        return gray
    angles = [(theta - np.pi/2) for rho,theta in lines[:,0]]
    angle = float(np.median(angles) * 180/np.pi)
    (h,w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def _local_mean_std(gray, ksize=61):
    mean = cv2.boxFilter(gray, ddepth=-1, ksize=(ksize, ksize))
    sqr  = cv2.boxFilter((gray*gray).astype(np.float32), ddepth=-1, ksize=(ksize, ksize))
    var  = np.maximum(sqr - mean.astype(np.float32)**2, 0.0)
    std  = np.sqrt(var)
    return mean.astype(np.float32), std

def binarize_suite(gray):
    imgs = []
    # Bradley
    mean = cv2.boxFilter(gray, ddepth=-1, ksize=(71,71))
    th_brad = (gray.astype(np.float32) <= (mean*(1-0.15))).astype(np.uint8)*255
    imgs.append(th_brad)
    # Niblack
    m, s = _local_mean_std(gray, 61)
    th_nib = (gray.astype(np.float32) > (m + (-0.2)*s)).astype(np.uint8)*255
    imgs.append(th_nib)
    # Sauvola
    R, k = 128.0, 0.5
    th_sau = (gray.astype(np.float32) > (m*(1 - k*(1 - s/R)))).astype(np.uint8)*255
    imgs.append(th_sau)
    # Wolf (approx)
    Mmin = cv2.erode(gray, np.ones((61,61), np.uint8)).astype(np.float32)
    Rmax = max(float(np.max(s)), 1e-3)
    k2 = 0.5
    wolf = ((1-k2)*m + k2*Mmin + k2*(s/Rmax)*(m - Mmin))
    th_wolf = (gray.astype(np.float32) > wolf).astype(np.uint8)*255
    imgs.append(th_wolf)
    return imgs

def pixel_vote(bin_imgs, majority=None):
    stack = np.stack([b//255 for b in bin_imgs], axis=0)
    maj = majority or (len(bin_imgs)//2 + 1)
    voted = (stack.sum(axis=0) >= maj).astype(np.uint8)*255
    k = np.ones((3,3), np.uint8)
    voted = cv2.morphologyEx(voted, cv2.MORPH_OPEN, k, iterations=1)
    voted = cv2.morphologyEx(voted, cv2.MORPH_CLOSE, k, iterations=1)
    return voted

def enhanced_preprocess(img_color):
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY) if len(img_color.shape)==3 else img_color
    gray = deskew(gray)
    if min(gray.shape[:2]) < 900:
        gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    bins = binarize_suite(gray)
    voted = pixel_vote(bins)
    return gray, voted

# Lightweight enhancement (used in ROI pass)
def enhance_image(img):
    if img is None:
        return None, None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g2 = clahe.apply(g)
    g3 = cv2.fastNlMeansDenoising(g2, h=7)
    th = cv2.adaptiveThreshold(g3, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 31, 15)
    return g3, th

def roi_candidates(img, limit: Optional[int]=None):
    H, W = img.shape[:2]
    rois = []
    def clamp(x0,y0,x1,y1):
        x0 = max(0, min(W-1, x0)); x1 = max(0, min(W-1, x1))
        y0 = max(0, min(H-1, y0)); y1 = max(0, min(H-1, y1))
        if x1<=x0 or y1<=y0: return None
        return (x0,y0,x1,y1)
    for frac in [0.30, 0.40, 0.50]:
        for band in [(0, 0, W, int(H*frac)),
                     (0, int(H*(1-frac)), W, H)]:
            b = clamp(*band)
            if b: x0,y0,x1,y1=b; rois.append(img[y0:y1, x0:x1])
    b3 = clamp(0, int(H*0.35), W, int(H*0.65))
    if b3: x0,y0,x1,y1=b3; rois.append(img[y0:y1, x0:x1])
    for frac in [0.30, 0.45]:
        b4 = clamp(0, 0, int(W*frac), H)
        b5 = clamp(int(W*(1-frac)), 0, W, H)
        if b4: x0,y0,x1,y1=b4; rois.append(img[y0:y1, x0:x1])
        if b5: x0,y0,x1,y1=b5; rois.append(img[y0:y1, x0:x1])
    if limit:
        return rois[:max(1, limit)]
    return rois

# =========================
# Multi-engine OCR + confusion map
# =========================

TRY_TESSERACT = True
try:
    import pytesseract
except Exception:
    TRY_TESSERACT = False

def ocr_tokens_google(img, pass_tag):
    bjpg = imencode_ext(".jpg", img)
    ocr = vision_ocr_image_bytes(bjpg, pass_tag)
    return extract_text_tokens(ocr), extract_full_text(ocr)

def ocr_tokens_tesseract(img):
    if not TRY_TESSERACT:
        return [], ""
    if len(img.shape)==2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cfg = "--psm 6"
    txt = pytesseract.image_to_string(rgb, config=cfg)
    tokens = re.split(r"\s+", txt.strip())
    return tokens, txt

CONFUSION_ALPHA2DIGIT = {'O':'0','D':'0','Q':'0','I':'1','l':'1','|':'1','Z':'2','S':'5','B':'8','G':'6'}
def normalize_code_with_confusion(s: str) -> str:
    return "".join(CONFUSION_ALPHA2DIGIT.get(ch, ch) for ch in s)

# =========================
# Date recognition & auto-correction
# =========================

MONTHS = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)"
MONTH_RE = re.compile(MONTHS, re.IGNORECASE)
DATE_SEP_RE = r"[\./\*\/]"  # treat / . * all as separators

def is_date_token(token: str) -> bool:
    if re.fullmatch(r"\d{1,2}"+DATE_SEP_RE+r"\d{1,2}", token):
        return True
    if MONTH_RE.fullmatch(token):
        return True
    return False

def _md_valid(mm: str, dd: str) -> bool:
    try:
        mi, di = int(mm), int(dd)
        return 1 <= mi <= 12 and 1 <= di <= 31
    except Exception:
        return False

def autocorrect_date_token(tok: str) -> Optional[str]:
    """
    Auto-correct common date misreads:
      - 67/25 -> 07/25   (month tens digit 6/9 misread for 0)
      - 97125 -> 07/25   (0->9 and separator '/' -> '1')
    Return normalized 'MM/DD' or None.
    """
    if not tok:
        return None
    s = tok.strip()

    # 1) mm<sep>dd
    m = re.match(rf"^(\d{{1,2}}){DATE_SEP_RE}(\d{{1,2}})$", s)
    if m:
        mm, dd = m.group(1), m.group(2)
        if _md_valid(mm, dd):
            return f"{int(mm):02d}/{int(dd):02d}"
        # month out of range → try fixing leading 6/9 to 0 (07/25 misread as 67/25 or 97/25)
        if len(mm) == 2 and mm[0] in "69" and _md_valid("0"+mm[1], dd):
            return f"0{mm[1]}/{int(dd):02d}"
        return None

    # 2) 5 digits where separator misread as '1'
    if re.fullmatch(r"\d{5}", s):
        cands = []
        for pos in (1, 2):
            if s[pos] != "1":
                continue
            mm, dd = s[:pos], s[pos+1:]
            mm2 = mm
            if len(mm2) == 2 and mm2[0] in "69":
                mm2 = "0" + mm2[1]
            if _md_valid(mm2, dd):
                cands.append(f"{int(mm2):02d}/{int(dd):02d}")
        if len(cands) == 1:
            return cands[0]
    return None

def normalize_digits(s: str) -> str:
    if not s: return s
    return re.sub(r"[^\d]", "", s)

def looks_like_date_misread_as_code5(s: str) -> bool:
    """If a 5-digit number can be corrected to valid mm/dd, treat it as a date misread."""
    if not s or not re.fullmatch(r"\d{5}", s):
        return False
    return autocorrect_date_token(s) is not None

# =========================
# Blacklist / parsing
# =========================

LEADING_NOISE_PREFIX = {"EVA", "VA"}  # avoid logos/material marks from leaking into names

def blacklist_check(word: str) -> str:
    names = ["the","foot","people","sydney","sydneycity","citypodiatry","sydneycitypodiatry","podiatry",
             "myfootdr","feet","townsville","clinic","performancepodiatry","my","orthotic","orthotics",
             "sport","sports","health","motion","active","group","central","lakes","bay","factory",
             "riverside","city","footpoint","gentle","achieve","runners","smart","orthoses","footsmart"]
    for n in names:
        if word == n:
            return ""
    return word

def _clean_name_tokens(tokens: List[str]) -> List[str]:
    cleaned = []
    for t in tokens:
        if not t:
            continue
        if t.isdigit() or is_date_token(t):
            continue
        pure = re.sub(r"[^A-Za-z.\-]", "", t)
        if not pure or len(re.sub(r"[^A-Za-z]", "", pure)) < 2:
            continue
        up = re.sub(r"[^A-Za-z]", "", pure).upper()
        if up in LEADING_NOISE_PREFIX:
            continue
        if blacklist_check(pure.lower()) == "":
            continue
        cleaned.append(pure)
    return cleaned

def standardize_clinic_name(s: str) -> str:
    if not s: return ""
    s = s.strip()
    s = re.sub(r"(\d{1,2}"+DATE_SEP_RE+r"\d{1,2})", "", s)
    s = re.sub(r"\d+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens_to_keywords(tokens: List[str]) -> List[str]:
    """
    Collect all tokens containing letters into a keyword bag; keep . / -; uppercase;
    drop pure digits and obvious dates.
    """
    kws = []
    for t in tokens:
        if not t: continue
        if t.isdigit():
            continue
        if is_date_token(t) or autocorrect_date_token(t):
            continue
        pure = re.sub(r"[^A-Za-z.\-]", "", t)
        if not re.search(r"[A-Za-z]", pure):
            continue
        up_letters = re.sub(r"[^A-Za-z]", "", pure).upper()
        if up_letters in LEADING_NOISE_PREFIX:
            continue
        kws.append(pure.upper())
    # dedupe but keep stable ordering (by length desc → lexicographic)
    kws = sorted(set(kws), key=lambda x:(-len(x), x))
    return kws

def merge_tokens_for_code(tokens: List[str]) -> Tuple[List[str], Optional[str], int]:
    n = len(tokens)
    for i in range(n):
        if i+1 < n and tokens[i].isdigit() and tokens[i+1].isdigit():
            merged = tokens[i] + tokens[i+1]
            if 5 <= len(merged) <= 6:
                new_tokens = tokens[:i] + [merged] + tokens[i+2:]
                return new_tokens, merged, i
        if i+2 < n and tokens[i].isdigit() and tokens[i+1].isdigit() and tokens[i+2].isdigit():
            merged = tokens[i] + tokens[i+1] + tokens[i+2]
            if 5 <= len(merged) <= 6:
                new_tokens = tokens[:i] + [merged] + tokens[i+3:]
                return new_tokens, merged, i
    for i, t in enumerate(tokens):
        t2 = normalize_digits(t)
        if re.fullmatch(r"\d{5,6}", t2):
            return tokens, t2, i
    return tokens, None, -1

def split_fields(tokens: List[str]):
    # Merge/extract code
    tokens, code, code_idx = merge_tokens_for_code(tokens)

    # Date (try auto-correction first)
    date = None; date_idx = -1; year = None
    for i, t in enumerate(tokens):
        fixed = autocorrect_date_token(t)
        if fixed:
            date = fixed; date_idx = i; break
    if date is None:
        for i, t in enumerate(tokens):
            if i < len(tokens)-1 and re.fullmatch(r"[A-Za-z]+", tokens[i]) and re.fullmatch(r"\d{1,2}", tokens[i+1]):
                md = f"{tokens[i]} {tokens[i+1]}"
                if re.fullmatch(MONTHS + r" \d{1,2}", md, re.IGNORECASE):
                    date = md; date_idx = i; break
            if re.fullmatch(r"\d{1,2}"+DATE_SEP_RE+r"\d{1,2}", t):
                fixed = autocorrect_date_token(t) or t
                date = fixed; date_idx = i; break
            if re.fullmatch(r"\d{4}", t):  # standalone year
                year = t

    # Name (allow '.' and '-')
    username = ""; username_idx = -1
    if code_idx > 0:
        head_tokens = []
        for i in range(0, code_idx):
            tok = tokens[i]
            if is_date_token(tok) or autocorrect_date_token(tok) or tok.isdigit():
                continue
            if re.fullmatch(r"[A-Za-z0-9.\-]+", tok):
                head_tokens.append(tok)
        head_tokens = _clean_name_tokens(head_tokens)
        if head_tokens:
            dot_names = [t for t in head_tokens if "." in t]
            username = "".join(dot_names) if dot_names else "".join(head_tokens)
            username_idx = max(0, code_idx - len(head_tokens))
    else:
        if len(tokens) >= 2 and all(re.fullmatch(r"[A-Za-z.\-]+", t) for t in tokens[:2]):
            cands = _clean_name_tokens([tokens[0], tokens[1]])
            username = " ".join(cands) if cands else ""
            username_idx = 0
        elif len(tokens) >= 1 and re.fullmatch(r"[A-Za-z.\-]+", tokens[0]):
            cands = _clean_name_tokens([tokens[0]])
            username = cands[0] if cands else ""
            username_idx = 0

    # Clinic (collect tail tokens with letters first; then standardize by stripping digits/dates)
    clinic = ""
    start_idx = max(code_idx, date_idx)
    if start_idx >= 0 and start_idx + 1 < len(tokens):
        tail_tokens = tokens[start_idx+1:]
    else:
        exclude_idx = set([username_idx]) if username_idx >= 0 else set()
        tail_tokens = [t for i,t in enumerate(tokens) if i not in exclude_idx and t]

    cand = []
    for t in tail_tokens:
        if not t or t.isdigit() or is_date_token(t) or autocorrect_date_token(t): continue
        if re.search(r"[A-Za-z]", t):
            # exclude obvious "name-like" tokens (with dot/hyphen and look like names)
            nm = re.sub(r"[^A-Za-z.\-]", "", t)
            if nm and ("." in nm or "-" in nm):
                letters = re.sub(r"[^A-Za-z]", "", nm)
                if len(letters) >= 2:
                    continue
            cand.append(t)
    clinic = standardize_clinic_name(" ".join(cand))

    # Keywords bag
    keywords = tokens_to_keywords(tokens)

    return code, username, date, clinic, year, keywords

# =========================
# Similarities / keyword similarity
# =========================

def text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a or "", b or "").ratio()

def name_is_generic(name: str) -> bool:
    if not name: return True
    pure = re.sub(r"[^A-Za-z]", "", name).upper()
    if len(pure) <= 2: return True
    tokens = re.split(r"[^A-Za-z]+", name.upper())
    GENERIC = {"THE","FOOT","CLINIC","PODIATRY","ORTHOTIC","ORTHOTICS","SPORT","SPORTS","FEET","FOOTPOINT","CENTRAL","CITY","HEALTH"}
    score = sum(1 for t in tokens if t in GENERIC and t)
    return score >= 2

def name_strict_ok(name: str) -> bool:
    if not name or name_is_generic(name): return False
    letters = re.sub(r"[^A-Za-z]", "", name)
    return len(letters) >= 2

def hamming_distance(a: str, b: str) -> Optional[int]:
    if not a or not b or len(a) != len(b): return None
    return sum(1 for x,y in zip(a,b) if x != y)

def keywords_similarity(a: List[str], b: List[str], token_thr: float = TOKEN_SIM_THR) -> float:
    """
    Soft Jaccard: for each token in a, find best unmatched token in b (≥ token_thr) as a hit; then do reverse;
    finally hits / union.
    """
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    A = list(a); B = list(b)
    matched_a = set(); matched_b = set()
    for i, ta in enumerate(A):
        best, jbest = 0.0, -1
        for j, tb in enumerate(B):
            if j in matched_b: continue
            sim = text_similarity(ta, tb)
            if sim > best:
                best, jbest = sim, j
        if best >= token_thr and jbest >= 0:
            matched_a.add(i); matched_b.add(jbest)
    hits = len(matched_a)
    for j, tb in enumerate(B):
        if j in matched_b: continue
        best = 0.0
        for i, ta in enumerate(A):
            if i in matched_a: continue
            sim = text_similarity(tb, ta)
            if sim > best: best = sim
        if best >= token_thr:
            hits += 1
    union = len(set(A)) + len(set(B)) - hits
    union = max(union, 1)
    return hits / union

# =========================
# Slots & policy (Blocking + pair guard + scoring + review band)
# =========================

@dataclass
class Slot:
    location_placed: bool = False
    pair_found: bool = False
    count: int = 0
    clinic: str = ""
    code: Optional[str] = None
    name: str = ""
    date: Optional[str] = None
    pass_tag: str = ""
    has_code: bool = False
    keywords: List[str] = None

def build_dictionary(n: int) -> Dict[int, Slot]:
    return {i: Slot(keywords=[]) for i in range(n)}

def block_keys_for(code: Optional[str], name: str, keywords: List[str]):
    keys = set()
    if code:
        keys.add(f"C:{code[:3]}")
        keys.add(f"L:{len(code)}")
    if name:
        pure = re.sub(r"[^A-Za-z]", "", name).upper()
        if len(pure)>=2:
            keys.add(f"N:{pure[:2]}")
    for kw in (keywords or []):
        letters = re.sub(r"[^A-Za-z]", "", kw).upper()
        if len(letters) >= 3:
            keys.add(f"K:{letters[:3]}")
    return keys

def _blocks_add(slot_id: int, code: Optional[str], name: str, keywords: List[str]):
    for k in block_keys_for(code, name, keywords):
        BLOCKS.setdefault(k, set()).add(slot_id)

def _candidate_slots_by_blocks(code, username, keywords):
    keys = block_keys_for(code, username, keywords)
    cand = set()
    for k in keys:
        cand |= BLOCKS.get(k, set())
    return [i for i in cand if 0 <= i < PENDING_START]

@dataclass
class PairDecision:
    idx: int
    file: str
    candidate_slot: Optional[int]
    chosen_slot: int
    decision: str
    name_sim: float
    kw_sim: float
    code_eq: bool
    code_hamming: Optional[int]
    pair_score: float
    reason: str

def log_pair_decision(pd: PairDecision):
    PAIR_LOGS.append({
        "idx": pd.idx,
        "file": pd.file,
        "candidate_slot": pd.candidate_slot if pd.candidate_slot is not None else -1,
        "chosen_slot": pd.chosen_slot,
        "decision": pd.decision,
        "name_sim": round(pd.name_sim, 4),
        "kw_sim": round(pd.kw_sim, 4),
        "code_eq": int(pd.code_eq),
        "code_hamming": pd.code_hamming if pd.code_hamming is not None else "",
        "pair_score": round(pd.pair_score, 4),
        "reason": pd.reason
    })
    # console log (compact)
    cs = f"{pd.code_hamming}" if pd.code_hamming is not None else "—"
    print(f"PAIR {pd.idx:03d} -> {pd.chosen_slot:02d} | decision={pd.decision} | "
          f"name_sim={pd.name_sim:.2f} | kw_sim={pd.kw_sim:.2f} | "
          f"code_eq={pd.code_eq} | code_hamming={cs} | pair_score={pd.pair_score:.2f} | reason={pd.reason}")

def open_new_normal_slot(code, username, date, clinic, pass_tag, keywords, diction) -> int:
    for i in range(0, PENDING_START):
        s = diction[i]
        if not s.location_placed:
            s.location_placed = True
            s.count = 1
            s.code = code
            s.has_code = bool(code)
            s.name = username
            s.date = date
            s.clinic = clinic if clinic else s.clinic
            s.pass_tag = pass_tag
            s.keywords = list(keywords or [])
            _blocks_add(i, s.code, s.name, s.keywords)
            return i
    # normal slots full → use Pending
    sid = _pending_slot_round_robin()
    ps = diction[sid]
    ps.location_placed = True
    if ps.count == 0:
        ps.pass_tag = pass_tag
        if not ps.clinic: ps.clinic = "[UNRECOGNIZED]"
    ps.count += 1
    return sid

def can_pair_slot(target: Slot, new_code: Optional[str], new_name: str, new_keywords: List[str]) -> bool:
    """Hard checks before placing into an existing slot."""
    name_sim = text_similarity(new_name, target.name or "") if (new_name and target.name) else 0.0
    kw_sim   = keywords_similarity(new_keywords or [], target.keywords or []) if (new_keywords or target.keywords) else 0.0
    if target.code and new_code:
        if target.code == new_code:
            return True
        ham = hamming_distance(target.code, new_code)
        if ham is not None and ham <= CODE_MAX_HAMMING and name_sim >= NAME_PAIR_SIM:
            return True
        return False
    if name_sim >= NAME_PAIR_SIM:
        return True
    if kw_sim >= KEYWORDS_PAIR_SIM:
        return True
    return False

def pair_score(target: Slot, new_code: Optional[str], new_name: str, new_clinic: str,
               new_date: Optional[str], new_keywords: List[str]) -> float:
    cs = 0.0
    if target.code and new_code:
        if target.code == new_code: cs = 1.0
        else:
            ham = hamming_distance(target.code, new_code)
            if ham is not None:
                if ham <= 1: cs = 0.8
                elif ham <= 2: cs = 0.6
    ns = text_similarity(new_name or "", target.name or "")
    ls = text_similarity(standardize_clinic_name(new_clinic or ""), standardize_clinic_name(target.clinic or ""))
    ds = 1.0 if (target.date and new_date and target.date == new_date) else 0.0
    ks = keywords_similarity(new_keywords or [], target.keywords or []) if (new_keywords or target.keywords) else 0.0
    return 1.0*cs + 0.6*ns + 0.2*ls + 0.1*ds + 0.4*ks

def diction_check_fill(code: Optional[str], username: str, date: Optional[str],
                       clinic: str, pass_tag: str, keywords: List[str],
                       diction: Dict[int, Slot], img_idx: int, file_name: str) -> int:
    """
    Slot selection + pair guard + scoring & review band:
      1) Find not-full slots with exact code match (priority);
      2) Else, use Blocking buckets to fetch name-strong (≥ NAME_STRICT_SIM) or keyword-strong (≥ KEYWORDS_PAIR_SIM) candidates
         and gate them via can_pair_slot();
      3) If hard checks fail: if pair_score falls into REVIEW_BAND, add to review list; then open a new slot;
      4) If no candidates: open a new slot; otherwise fall back to Pending.
    """
    # 1) exact code match (priority)
    if code:
        cand = [i for i in _candidate_slots_by_blocks(code, username, keywords)
                if diction[i].location_placed and diction[i].count < 2 and diction[i].code == code]
        if not cand:
            cand = [i for i,s in diction.items()
                    if 0 <= i < PENDING_START and s.location_placed and s.count < 2 and s.code == code]
        if cand:
            best_i = max(cand, key=lambda i: text_similarity(username, diction[i].name or "") if (username and diction[i].name) else 0.0)
            tgt = diction[best_i]
            name_sim = text_similarity(username or "", tgt.name or "")
            kw_sim   = keywords_similarity(keywords or [], tgt.keywords or [])
            ham = hamming_distance(tgt.code, code)
            reason = "code_exact" if tgt.code == code else f"code_hamming<={CODE_MAX_HAMMING} & name>={NAME_PAIR_SIM}"
            if can_pair_slot(tgt, code, username, keywords):
                sc = pair_score(tgt, code, username, clinic, date, keywords)
                tgt.pair_found = True; tgt.count += 1
                if not tgt.date and date:   tgt.date = date
                if (not tgt.clinic or tgt.clinic.startswith("[UNRECOGNIZED]")) and clinic: tgt.clinic = clinic
                if not tgt.keywords and keywords: tgt.keywords = list(keywords)
                _blocks_add(best_i, tgt.code, tgt.name, tgt.keywords)
                log_pair_decision(PairDecision(img_idx, file_name, best_i, best_i, "PAIR", name_sim, kw_sim, tgt.code==code, ham, sc, reason))
                return best_i
            # hard check failed → open new slot
            sc = pair_score(tgt, code, username, clinic, date, keywords)
            if REVIEW_BAND_LOW <= sc <= REVIEW_BAND_HIGH:
                REVIEW_ROWS.append((best_i, code, username, date, clinic, sc))
            new_slot = open_new_normal_slot(code, username, date, clinic, pass_tag, keywords, diction)
            log_pair_decision(PairDecision(img_idx, file_name, best_i, new_slot, "NEW_SLOT",
                                           name_sim, kw_sim, tgt.code==code, ham, sc, "guard_reject"))
            return new_slot

    # 2) search by name/keywords in Blocking buckets
    best_i, best_sim = -1, 0.0
    cand = _candidate_slots_by_blocks(code, username, keywords)
    if not cand:
        cand = [i for i,s in diction.items() if 0 <= i < PENDING_START and s.location_placed and s.count < 2 and (s.name or s.keywords)]
    for i in cand:
        s = diction[i]
        sim = text_similarity(username or "", s.name or "")
        if sim > best_sim:
            best_sim, best_i = sim, i

    reason = ""
    if best_i < 0 or best_sim < NAME_STRICT_SIM:
        best_i2, best_ks = -1, 0.0
        for i in cand:
            s = diction[i]
            ks = keywords_similarity(keywords or [], s.keywords or [])
            if ks > best_ks:
                best_ks, best_i2 = ks, i
        if best_i2 >= 0 and best_ks >= KEYWORDS_PAIR_SIM:
            best_i, best_sim = best_i2, best_ks
            reason = "kw>=thr"
    else:
        reason = "name>=thr"

    if best_i >= 0:
        tgt = diction[best_i]
        name_sim = text_similarity(username or "", tgt.name or "")
        kw_sim   = keywords_similarity(keywords or [], tgt.keywords or [])
        ham      = hamming_distance(tgt.code, code) if (tgt.code and code) else None
        if can_pair_slot(tgt, code, username, keywords):
            sc = pair_score(tgt, code, username, clinic, date, keywords)
            tgt.pair_found = True; tgt.count += 1
            if not tgt.code and code:   tgt.code = code; tgt.has_code = bool(code)
            if not tgt.date and date:   tgt.date = date
            if (not tgt.clinic or tgt.clinic.startswith("[UNRECOGNIZED]")) and clinic: tgt.clinic = clinic
            if not tgt.keywords and keywords: tgt.keywords = list(keywords)
            _blocks_add(best_i, tgt.code, tgt.name, tgt.keywords)
            log_pair_decision(PairDecision(img_idx, file_name, best_i, best_i, "PAIR", name_sim, kw_sim, (tgt.code==code if (tgt.code and code) else False), ham, sc, reason))
            return best_i
        sc = pair_score(tgt, code, username, clinic, date, keywords)
        if REVIEW_BAND_LOW <= sc <= REVIEW_BAND_HIGH:
            REVIEW_ROWS.append((best_i, code, username, date, clinic, sc))
        new_slot = open_new_normal_slot(code, username, date, clinic, pass_tag, keywords, diction)
        log_pair_decision(PairDecision(img_idx, file_name, best_i, new_slot, "NEW_SLOT",
                                       name_sim, kw_sim, (tgt.code==code if (tgt.code and code) else False), ham, sc, "guard_reject"))
        return new_slot

    # 3) open a new slot / Pending
    decision = "NEW_SLOT"
    slot = None
    if code or (username and name_strict_ok(username)) or (keywords and len(keywords) >= 1):
        slot = open_new_normal_slot(code, username, date, clinic, pass_tag, keywords, diction)
    else:
        sid = _pending_slot_round_robin()
        ps = diction[sid]
        ps.location_placed = True
        if ps.count == 0:
            ps.pass_tag = pass_tag
            if not ps.clinic: ps.clinic = "[UNRECOGNIZED]"
        ps.count += 1
        slot = sid
        decision = "PENDING"
    log_pair_decision(PairDecision(img_idx, file_name, None, slot, decision, 0.0, 0.0, False, None, 0.0, "no_candidate"))
    return slot

# =========================
# Parse one image (rotation channels & 6-digit voting; do not early-stop on 5-digit date-like)
# =========================

@dataclass
class ParseResult:
    code: Optional[str]
    username: str
    date: Optional[str]
    clinic: str
    pass_tag: str
    full_text: str
    img_path: str
    file_name: str
    keywords: List[str]

def ocr_ensemble_tokens(img, pass_tag="ens"):
    tok_all, full_all = [], []
    t1, f1 = ocr_tokens_google(img, pass_tag+"-g"); tok_all.append(t1); full_all.append(f1)
    if ENABLE_OCR_ENSEMBLE and TRY_TESSERACT:
        t2, f2 = ocr_tokens_tesseract(img); tok_all.append(t2); full_all.append(f2)
    flat = [t for L in tok_all for t in L]
    codes = []
    for t in flat:
        t2 = normalize_digits(normalize_code_with_confusion(t))
        if re.fullmatch(r"\d{5,6}", t2):
            codes.append(t2)
    code = None
    if codes:
        from collections import Counter
        c = Counter(codes)
        six = [k for k in c if len(k)==6]
        if six:
            code = max(six, key=lambda k:(c[k], k))
        else:
            code = max(c.keys(), key=lambda k:(c[k], k))
    name_tokens = [re.sub(r"[^A-Za-z.\-]","", t) for t in flat]
    name_tokens = [t for t in name_tokens if len(re.sub(r"[^A-Za-z]","",t))>=2]
    username = max(name_tokens, key=len) if name_tokens else ""
    full = "\n\n---\n".join(full_all)
    return code, username, full

def parse_tokens(tokens: List[str]) -> Tuple[Optional[str], str, Optional[str], str, Optional[str], List[str]]:
    code, username, date, clinic, year, keywords = split_fields(tokens)
    if code:
        code = normalize_digits(normalize_code_with_confusion(code))
    return code, (username or ""), date, (clinic or ""), year, keywords

def _rotate90(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
def _rotate270(img):
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

def parse_one_image(img_path: str, file_name: str, counters: dict) -> ParseResult:
    img = imread_unicode(img_path, cv2.IMREAD_COLOR)
    if img is None:
        counters["read_fail"] = counters.get("read_fail", 0) + 1
        return ParseResult(None, "", None, "", "orig", "", img_path, file_name, [])

    passes = []

    # Pass-0: pixel-voting preprocess (priority)
    if ENABLE_VOTE_PREPROC:
        try:
            gray, voted = enhanced_preprocess(img)
            passes.append(("vote", voted))
        except Exception:
            pass

    # Pass-1: original
    passes.append(("orig", img))

    # Pass-2: light enhancement
    g3, th = enhance_image(img)
    if g3 is not None:
        passes.append(("enhanced", g3))
        counters["enhanced_used"] = counters.get("enhanced_used", 0) + 1

    # Pass-3: ±90° rotations
    try:
        passes.append(("rot90", _rotate90(img)))
        passes.append(("rot270", _rotate270(img)))
    except Exception:
        pass

    # Pass-4: ROI
    rois = roi_candidates(img, limit=HEAVY_ROI_MAX)
    if rois:
        counters["roi_used"] = counters.get("roi_used", 0) + len(rois)
    for idx, roi in enumerate(rois):
        passes.append((f"roi_{idx}", roi))

    # Pass-5: thresholded image
    if th is not None:
        passes.append(("th", th))

    # Aggregate: no early stop; collect all channel candidates then decide
    code_votes = {}
    best_name, best_date, best_clinic, best_keywords, best_pass = "", None, "", [], "orig"
    full_text_any = ""

    for tag, im in passes:
        tk, full = ocr_tokens_from_img(im, tag)
        c, u, d, cl, yr, kws = parse_tokens(tk)
        full_text_any = full_text_any or full

        # 6-digit code voting; 5-digit code treated carefully if date-like
        if c:
            if len(c) == 6:
                code_votes[c] = code_votes.get(c, 0) + 1
            elif len(c) == 5:
                if not looks_like_date_misread_as_code5(c):
                    code_votes[c] = code_votes.get(c, 0) + 1

        # update best name/clinic/date/keywords
        if (not best_name and u) or (u and len(u) > len(best_name)):
            up = re.sub(r"[^A-Za-z]", "", u).upper()
            if up in LEADING_NOISE_PREFIX:
                pass
            else:
                best_name = u; best_pass = tag
        if not best_date and d:
            best_date = d
        # derive date from a 5-digit "code" if it's actually a date
        if not best_date and c and len(c)==5:
            fd = autocorrect_date_token(c)
            if fd: best_date = fd
        if (not best_clinic and cl) or (cl and len(cl) > len(best_clinic)):
            best_clinic = cl
        if not best_keywords and kws:
            best_keywords = kws

    # choose final code: prefer 6-digit with highest votes
    final_code = None
    if code_votes:
        six = {k:v for k,v in code_votes.items() if len(k)==6}
        if six:
            final_code = max(six.keys(), key=lambda k:(six[k], k))
        else:
            final_code = max(code_votes.keys(), key=lambda k:(code_votes[k], k))

    return ParseResult(final_code, best_name, best_date, best_clinic, best_pass, full_text_any, img_path, file_name, best_keywords)

# =========================
# Light code recheck (in-image) + external fill (vs seen 6-digit codes)
# =========================

def _pending_slot_round_robin() -> int:
    global _PENDING_TOGGLE
    sid = PENDING_IDS[_PENDING_TOGGLE % 2]
    _PENDING_TOGGLE += 1
    return sid

def edit_distance_insert1(code5: str, code6: str) -> bool:
    if len(code5) != 5 or len(code6) != 6: return False
    i = j = 0; mismatch = 0
    while i < 5 and j < 6:
        if code5[i] == code6[j]:
            i += 1; j += 1
        else:
            j += 1; mismatch += 1
            if mismatch > 1: return False
    return True

def light_code_recheck(pr: ParseResult, counters: dict, seen_codes_6: set) -> Optional[str]:
    """
    Only when a 5-digit code is found:
      - rescan with ROI/enhancement; if still 5-digit, try "insert-one" match against seen 6-digit codes.
      - this fix applies to current image only (no cross-slot merging).
    """
    if not pr.code:
        return None
    code = normalize_digits(pr.code)
    if len(code) != 5:
        return None
    if looks_like_date_misread_as_code5(code):
        return None

    img = imread_unicode(pr.img_path, cv2.IMREAD_COLOR)
    if img is None:
        return None

    cand = [code]

    rois = roi_candidates(img, limit=LIGHT_ROI_MAX)
    if rois:
        counters["code_light_roi"] = counters.get("code_light_roi", 0) + len(rois)
    for idx, roi in enumerate(rois):
        tk, _ = ocr_tokens_from_img(roi, f"code_light_roi_{idx}")
        for t in tk:
            t2 = normalize_digits(normalize_code_with_confusion(t))
            if re.fullmatch(r"\d{5,6}", t2):
                if len(t2)==5 and looks_like_date_misread_as_code5(t2):
                    continue
                cand.append(t2)

    g3, _ = enhance_image(img)
    if g3 is not None:
        counters["code_light_enh"] = counters.get("code_light_enh", 0) + 1
        tk2, _ = ocr_tokens_from_img(g3, "code_light_enh")
        for t in tk2:
            t2 = normalize_digits(normalize_code_with_confusion(t))
            if re.fullmatch(r"\d{5,6}", t2):
                if len(t2)==5 and looks_like_date_misread_as_code5(t2):
                    continue
                cand.append(t2)

    six_internal = [c for c in cand if len(c) == 6]
    chosen = six_internal[0] if six_internal else None

    matches = [c6 for c6 in seen_codes_6 if edit_distance_insert1(code, c6)]
    if len(matches) == 1:
        chosen = matches[0]

    if chosen and chosen != code:
        counters["code_light_fix"] = counters.get("code_light_fix", 0) + 1
        return chosen
    return None

# =========================
# Post-pass merge (strict: also gated by pair guard)
# =========================

def do_postpass_merge(diction: Dict[int, Slot], mode: str = "off", min_score: float = 0.84) -> int:
    if mode not in ("soft","strict"):
        return 0

    merged = 0
    normals = [i for i,s in diction.items() if s.location_placed and (0 <= i < PENDING_START)]
    pendings = [i for i in PENDING_IDS if diction[i].location_placed]

    for psid in pendings:
        ps = diction[psid]
        if not ps.location_placed or ps.count != 1:
            continue

        pname = ps.name or ""; pclin = ps.clinic or ""
        pkeys = ps.keywords or []
        best_sid, best_score, best_name_sim, best_clinic_sim, best_kw_sim = None, 0.0, 0.0, 0.0, 0.0

        for nsid in normals:
            ns = diction[nsid]
            if ns.count >= 2: continue
            name_sim = text_similarity(pname, ns.name or "") if (pname and ns.name) else 0.0
            clin_sim = text_similarity(pclin, ns.clinic or "") if (pclin and ns.clinic) else 0.0
            kw_sim   = keywords_similarity(pkeys, ns.keywords or []) if (pkeys or ns.keywords) else 0.0
            score = max(name_sim, kw_sim, 0.9*clin_sim)
            if score > best_score:
                best_sid, best_score = nsid, score
                best_name_sim, best_clinic_sim, best_kw_sim = name_sim, clin_sim, kw_sim

        allow = False
        if best_sid is not None:
            if mode == "soft":
                allow = best_score >= min_score
            else:
                allow = (best_name_sim >= 0.88) or (best_kw_sim >= 0.84) or (best_name_sim >= 0.80 and best_clinic_sim >= 0.80)

        if allow and best_sid is not None:
            tgt = diction[best_sid]
            if not can_pair_slot(tgt, ps.code, ps.name, ps.keywords or []):
                continue

            tgt.count = min(2, tgt.count + ps.count)
            if not tgt.code and ps.code:
                tgt.code = ps.code; tgt.has_code = True
            if not tgt.name and ps.name:
                tgt.name = ps.name
            if not tgt.date and ps.date:
                tgt.date = ps.date
            if (not tgt.clinic or tgt.clinic.startswith("[UNRECOGNIZED]")) and ps.clinic:
                tgt.clinic = ps.clinic
            if not tgt.keywords and ps.keywords:
                tgt.keywords = list(ps.keywords)

            _blocks_add(best_sid, tgt.code, tgt.name, tgt.keywords)

            ps.location_placed = False
            ps.count = 0
            ps.code = None
            ps.name = ""
            ps.date = None
            ps.clinic = ""
            ps.has_code = False
            ps.keywords = []
            merged += 1

    return merged

# =========================
# Main flow
# =========================

def run_dir(image_dir: str, limit: int = 0):
    files = sorted([f for f in os.listdir(image_dir)
                    if os.path.splitext(f)[1].lower() in (".jpg",".jpeg",".png",".bmp",".webp")])
    if limit and limit > 0:
        files = files[:limit]

    print(f"🖼️ Images to process: {len(files)} | dir: {image_dir}")

    diction = build_dictionary(MAX_PAIR_NUMBER)
    BLOCKS.clear()
    REVIEW_ROWS.clear()
    PAIR_LOGS.clear()

    counters = {
        "enhanced_used": 0, "roi_used": 0,
        "enhanced_recover_code": 0, "roi_recover_code": 0, "th_recover_code": 0,
        "read_fail": 0
    }

    per_image_logs = []
    t0_all = time.perf_counter()

    seen_codes_6 = set()

    for idx, fname in enumerate(files):
        fpath = os.path.join(image_dir, fname)
        t0 = time.perf_counter()
        try:
            pr = parse_one_image(fpath, fname, counters)

            # 5→6 light recheck (current image only)
            if pr.code:
                fixed = light_code_recheck(pr, counters, seen_codes_6)
                if fixed and fixed != pr.code and len(fixed)==6:
                    pr = ParseResult(fixed, pr.username, pr.date, pr.clinic,
                                     pr.pass_tag, pr.full_text, pr.img_path, pr.file_name, pr.keywords)

            # choose slot + hard checks (with pairing logs)
            slot = diction_check_fill(pr.code, pr.username, pr.date, pr.clinic, pr.pass_tag, pr.keywords, diction, idx, fname)
            if slot < 0: slot = 0

            item = diction[slot]
            # fill fields for normal slots
            if 0 <= slot < PENDING_START:
                if item.count == 1 and not item.name:
                    item.name   = pr.username
                    _blocks_add(slot, item.code, item.name, item.keywords)
                if item.count == 1 and not item.code:
                    item.code   = pr.code; item.has_code = bool(pr.code)
                    _blocks_add(slot, item.code, item.name, item.keywords)
                if item.count == 1 and not item.date:   item.date   = pr.date
                if item.count == 1 and (not item.clinic or item.clinic.startswith("[UNRECOGNIZED]")):
                    item.clinic = pr.clinic
                if item.count == 1 and (not item.keywords):
                    item.keywords = list(pr.keywords or [])
                if item.count == 1 and not item.pass_tag:
                    item.pass_tag = pr.pass_tag

                if item.code and len(item.code) == 6:
                    seen_codes_6.add(item.code)

            elapsed_ms = int((time.perf_counter() - t0)*1000)

            # new-style log: include OCR token list
            print(f"→ placed into slot {slot} | code={pr.code} name={pr.username} date={pr.date} clinic={pr.clinic} | count={item.count}")
            print(f"[{idx:03d}] slot={slot:2d} | code={pr.code or 'None'} | OCR tokens:{fmt_kw_list(pr.keywords)} | "
                  f"name={pr.username or ''} | date={pr.date or 'None'} | file={fname} | {elapsed_ms} ms | pass={pr.pass_tag}")

            per_image_logs.append({
                "idx": idx, "slot": slot, "file": fname,
                "code": pr.code, "name": pr.username, "date": pr.date, "clinic": pr.clinic,
                "keywords": pr.keywords,
                "elapsed_ms": elapsed_ms, "pass": pr.pass_tag
            })

        except Exception as e:
            print(f"[{idx:03d}] ERROR file={fname} :: {e}")
            per_image_logs.append({"idx": idx, "slot": -1, "file": fname, "error": str(e)})

    merged_pending = do_postpass_merge(diction, POSTPASS_MODE, POSTPASS_MIN_SCORE)
    if merged_pending > 0:
        print(f"\n===== Post-pass merge enabled ({POSTPASS_MODE}) =====")
        print(f"Merged pending slots: {merged_pending}")

    used_normal   = [i for i,s in diction.items() if s.location_placed and (0 <= i < PENDING_START)]
    paired_normal = [i for i in used_normal if diction[i].count == 2]
    single_normal = [i for i in used_normal if diction[i].count == 1]
    pending_slots = [i for i in PENDING_IDS if diction[i].location_placed]
    pending_count = sum(diction[i].count for i in pending_slots)
    code_slots    = [i for i in range(MAX_PAIR_NUMBER) if diction[i].location_placed and diction[i].has_code]
    nocode_slots  = [i for i in range(MAX_PAIR_NUMBER) if diction[i].location_placed and not diction[i].has_code]

    print("\n===== Summary (v9.0-keywords+autocorrect+pairlog) =====")
    print(f"Total images: {len(files)}")
    print(f"Used normal slots: {len(used_normal)} (0..67, at most 2 items per slot)")
    print(f"Paired normal slots (count=2): {len(paired_normal)}")
    print(f"Single normal slots (count=1): {len(single_normal)}")
    print(f"Pending slots: {len(pending_slots)} (fixed 68/69), pending item count: {pending_count}")
    print(f"Slots with code: {len(code_slots)}, without code: {len(nocode_slots)}")
    print(f"Enhanced used: {counters.get('enhanced_used',0)} | ROI used: {counters.get('roi_used',0)} | "
          f"Enhanced code recovery: {counters.get('enhanced_recover_code',0)} | ROI code recovery: {counters.get('roi_recover_code',0)} | "
          f"TH code recovery: {counters.get('th_recover_code',0)}")
    print(f"Code light recheck: ROI {counters.get('code_light_roi',0)} | Enh {counters.get('code_light_enh',0)} | Fixed {counters.get('code_light_fix',0)}")

    # Exports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = os.path.join(SAVE_DIR, f"test_results_slots_{timestamp}.csv")
    json_path = os.path.join(SAVE_DIR, f"test_results_diction_{timestamp}.json")
    log_path  = os.path.join(SAVE_DIR, f"test_results_log_{timestamp}.txt")
    pairs_csv = os.path.join(SAVE_DIR, f"pairs_log_{timestamp}.csv")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["slot","count","code","name","date","clinic","keywords","pass","has_code","is_pending"])
        for i in range(MAX_PAIR_NUMBER):
            s = diction[i]
            kws = " ".join(s.keywords or [])
            w.writerow([i, s.count, s.code or "", s.name, s.date or "", s.clinic, kws, s.pass_tag, int(s.has_code), int(i in PENDING_IDS and s.location_placed)])

    snap = {i: asdict(diction[i]) for i in range(MAX_PAIR_NUMBER)}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2)

    with open(log_path, "w", encoding="utf-8") as f:
        for r in per_image_logs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Pairing details
    if PAIR_LOGS:
        with open(pairs_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(PAIR_LOGS[0].keys()))
            w.writeheader()
            w.writerows(PAIR_LOGS)
        print(f"Pairing details: {pairs_csv}")

    # Review list
    if REVIEW_ROWS:
        rv_path = os.path.join(SAVE_DIR, f"review_{timestamp}.csv")
        with open(rv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(["slot","new_code","new_name","new_date","new_clinic","score"])
            w.writerows(REVIEW_ROWS)
        print(f"Samples requiring manual review: {rv_path}")

    print(f"\nExported slot details: {csv_path}")
    print(f"Exported dictionary snapshot: {json_path}")
    print(f"Light log: {log_path}")
    print(f"Total time: {int((time.perf_counter()-t0_all))} s")

# ============ CLI ============
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Batch Image OCR + Slot Pairing (v9.0: keywords+autocorrect+pairlog)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dir", type=str, default=TEST_IMAGE_DIR, help="Directory of images")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N images (0=all)")
    parser.add_argument("--postpass", type=str, choices=["off","soft","strict"], default=POSTPASS_MODE, help="Post-pass merge mode")
    parser.add_argument("--min-score", type=float, default=POSTPASS_MIN_SCORE, help="Minimum similarity for soft mode")
    parser.add_argument("--no-cache", action="store_true", help="Disable OCR cache")
    parser.add_argument("--light-roi", type=int, default=LIGHT_ROI_MAX, help="Number of ROIs for code light recheck")
    parser.add_argument("--heavy-roi", type=int, default=HEAVY_ROI_MAX, help="Number of ROIs for no-code deep scan")
    # Key thresholds
    parser.add_argument("--name-strict", type=float, default=NAME_STRICT_SIM, help="Strong name-candidate threshold")
    parser.add_argument("--name-pair", type=float, default=NAME_PAIR_SIM, help="Same-slot name threshold (hard check)")
    parser.add_argument("--code-maxdiff", type=int, default=CODE_MAX_HAMMING, help="Max allowed digit difference for codes")
    # Keyword bag thresholds
    parser.add_argument("--token-thr", type=float, default=TOKEN_SIM_THR, help="Per-token tolerance threshold")
    parser.add_argument("--kw-pair", type=float, default=KEYWORDS_PAIR_SIM, help="Keyword similarity threshold (hard check)")
    # Feature toggles
    parser.add_argument("--no-vote", action="store_true", help="Disable deskew+multi-binarization pixel-voting pass")
    parser.add_argument("--no-ensemble", action="store_true", help="Disable multi-engine OCR (use Google only)")
    # Review band
    parser.add_argument("--review-low", type=float, default=REVIEW_BAND_LOW, help="Review band lower bound")
    parser.add_argument("--review-high", type=float, default=REVIEW_BAND_HIGH, help="Review band upper bound")

    args = parser.parse_args()

    USE_CACHE = not args.no_cache
    POSTPASS_MODE = args.postpass
    POSTPASS_MIN_SCORE = float(args.min_score)
    LIGHT_ROI_MAX = int(args.light_roi)
    HEAVY_ROI_MAX = int(args.heavy_roi)
    NAME_STRICT_SIM = float(args.name_strict)
    NAME_PAIR_SIM = float(args.name_pair)
    CODE_MAX_HAMMING = int(args.code_maxdiff)
    TOKEN_SIM_THR = float(args.token_thr)
    KEYWORDS_PAIR_SIM = float(args.kw_pair)
    ENABLE_VOTE_PREPROC = not args.no_vote
    ENABLE_OCR_ENSEMBLE = not args.no_ensemble
    REVIEW_BAND_LOW = float(args.review_low)
    REVIEW_BAND_HIGH = float(args.review_high)

    if not os.path.isdir(args.dir):
        print(f"Directory not found: {args.dir}")
        sys.exit(1)

    run_dir(args.dir, limit=args.limit)
