"""Title cleaning, text-validity criterion, class-name lexicon, keyword rule (K0), site names.

All rules here are fixed by hand from the class list and from TRAIN titles only (proposal 2.3, 3.3 item 6);
nothing is fitted on validation or test data.

* clean_title(t)      HTML-unescaped, whitespace-collapsed title. This is what the text encoders see
                      (DistilBERT uncased lower-cases it itself).
* normalize(t)        clean_title + accents removed, lower-case, "&" -> "and", punctuation -> space.
                      Used for matching only (lexicon, validity, duplicate titles).
* is_valid_text(t)    pair-validity criterion 2: at least one content word (>= 3 letters, not in GENERIC_WORDS).
* ClassLexicon        class-name variants (name, singular/plural, a few hand-written synonyms) with
                      word-boundary matching; longest variant wins at a position.
* keyword_rule        K0: predict the most-mentioned class (ties: first mention), else a fallback class.
* site_name(t)        trailing "| Site" / "- Site" / "on Pinterest" segment, the only proxy for the source site.
"""
from __future__ import annotations

import html
import re
import unicodedata

# Words that carry no information about the dish: web boilerplate, generic recipe vocabulary, stop words.
GENERIC_WORDS = frozenset("""
recipe recipes ingredient ingredients food foods foodie home homepage blog blogs blogspot pinterest facebook
twitter instagram youtube flickr tumblr wordpress com www http https html htm php asp net org page pages index
the and for with how you your our from this that are was can all not its it's has have get one two
best easy perfect delicious homemade quick simple tasty yummy great good amazing awesome ultimate classic
cooking cook cooks kitchen kitchens menu menus dish dishes meal meals dinner dinners lunch snack snacks
photo photos image images picture pictures pic pics video videos search results result untitled welcome
new free online eat eats eating taste tastes make making made style day week weekend
""".split())

# Hand-written variants that the automatic singular/plural forms do not cover. Every variant maps to a
# single class (checked in ClassLexicon), so a mention is never ambiguous.
SYNONYMS = {
    "baby_back_ribs": ["back ribs", "baby back rib"],
    "beef_carpaccio": ["carpaccio"],
    "beef_tartare": ["beef tartar", "steak tartare", "steak tartar"],
    "beet_salad": ["beetroot salad"],
    "bibimbap": ["bi bim bap"],
    "bruschetta": ["bruschette"],
    "caesar_salad": ["cesar salad"],
    "cannoli": ["cannolo"],
    "caprese_salad": ["caprese"],
    "ceviche": ["cebiche", "seviche"],
    "cheese_plate": ["cheese platter", "cheese board"],
    "cheesecake": ["cheese cake", "cheese cakes"],
    "chicken_wings": ["buffalo wings", "hot wings"],
    "clam_chowder": ["chowder"],
    "crab_cakes": ["crabcake", "crabcakes"],
    "croque_madame": ["croque madam"],
    "cup_cakes": ["cupcake", "cupcakes", "cup cake"],
    "deviled_eggs": ["devilled eggs", "deviled egg", "devilled egg"],
    "donuts": ["donut", "doughnut", "doughnuts"],
    "eggs_benedict": ["egg benedict"],
    "falafel": ["felafel"],
    "filet_mignon": ["fillet mignon"],
    "fish_and_chips": ["fish n chips", "fish chips"],
    "french_fries": ["fries"],
    "french_onion_soup": ["onion soup"],
    "fried_calamari": ["calamari"],
    "frozen_yogurt": ["frozen yoghurt", "froyo"],
    "grilled_cheese_sandwich": ["grilled cheese"],
    "guacamole": ["guac"],
    "hamburger": ["burger", "burgers", "cheeseburger", "cheeseburgers"],
    "hot_and_sour_soup": ["hot sour soup", "hot n sour soup"],
    "hot_dog": ["hotdog", "hotdogs"],
    "huevos_rancheros": ["huevos ranchero"],
    "hummus": ["houmous", "hummous"],
    "ice_cream": ["icecream"],
    "lasagna": ["lasagne"],
    "lobster_roll_sandwich": ["lobster roll", "lobster rolls"],
    "macaroni_and_cheese": ["mac and cheese", "mac n cheese", "macaroni cheese", "mac cheese"],
    "macarons": ["macaron"],
    "omelette": ["omelet", "omelets"],
    "pad_thai": ["phad thai", "padthai"],
    "panna_cotta": ["pannacotta"],
    "peking_duck": ["beijing duck"],
    "pulled_pork_sandwich": ["pulled pork"],
    "red_velvet_cake": ["red velvet"],
    "shrimp_and_grits": ["shrimp n grits", "shrimp grits"],
    "spaghetti_bolognese": ["bolognese", "spag bol"],
    "spaghetti_carbonara": ["carbonara"],
    "strawberry_shortcake": ["shortcake", "shortcakes"],
    "tuna_tartare": ["tuna tartar", "ahi tartare"],
}

_SITE_SEP = re.compile(r"\s+(?:\||-|–|—|:|::|»|·|•)\s+")


def clean_title(text) -> str:
    """HTML entities decoded (twice: the crawl contains double-escaped '&amp;#039;'), whitespace collapsed."""
    t = "" if text is None else str(text)
    t = html.unescape(html.unescape(t))
    return re.sub(r"\s+", " ", t).strip()


def normalize(text) -> str:
    t = unicodedata.normalize("NFKD", clean_title(text))
    t = "".join(ch for ch in t if not unicodedata.combining(ch)).lower().replace("&", " and ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", t)).strip()


def content_words(text) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", normalize(text)) if len(w) >= 3 and w not in GENERIC_WORDS]


def is_valid_text(text) -> bool:
    """Pair-validity criterion 2 (proposal 2.3)."""
    return len(content_words(text)) > 0


def site_name(text) -> str:
    """Trailing site segment, e.g. 'Easy Apple Pie | Taste of Home' -> 'taste of home'; '' if none."""
    t = clean_title(text)
    m = re.search(r"\s+on\s+pinterest\s*$", t, flags=re.I)
    if m:
        return "pinterest"
    parts = _SITE_SEP.split(t)
    if len(parts) >= 2:
        last = normalize(parts[-1])
        if last and len(last.split()) <= 5:
            return last
    return ""


class ClassLexicon:
    """Class-name variants -> class, matched on normalize(text) with word boundaries."""

    def __init__(self, classes, synonyms: dict | None = None):
        self.classes = list(classes)
        synonyms = SYNONYMS if synonyms is None else synonyms
        variant_to_class: dict[str, str] = {}
        for c in self.classes:
            base = normalize(c.replace("_", " "))
            forms = {base, base + "s"}
            if base.endswith("s"):
                forms.add(base[:-1])
            if base.endswith(("ch", "sh", "x")):
                forms.add(base + "es")
            forms.update(normalize(s) for s in synonyms.get(c, []))
            for f in forms:
                if f in variant_to_class and variant_to_class[f] != c:
                    raise ValueError(f"lexicon variant '{f}' maps to both {variant_to_class[f]} and {c}")
                variant_to_class[f] = c
        self.variant_to_class = variant_to_class
        alts = sorted(variant_to_class, key=len, reverse=True)          # longest variant first
        self.pattern = re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(a) for a in alts) + r")(?![a-z0-9])")
        self.exact = {c: normalize(c.replace("_", " ")) for c in self.classes}

    def mentions(self, text) -> list[tuple[str, int]]:
        """[(class, position)] of every non-overlapping mention, left to right."""
        return [(self.variant_to_class[m.group(1)], m.start()) for m in self.pattern.finditer(normalize(text))]

    def mentioned_classes(self, text) -> set[str]:
        return {c for c, _ in self.mentions(text)}

    def mentions_exact(self, text, cls: str) -> bool:
        """Exact class name (underscores -> spaces) as a word sequence: the proposal's 73 % statistic."""
        return re.search(r"(?<![a-z0-9])" + re.escape(self.exact[cls]) + r"(?![a-z0-9])", normalize(text)) is not None

    def keyword_rule(self, text):
        """K0: the most-mentioned class; ties -> the class mentioned first. None if no class is mentioned."""
        ms = self.mentions(text)
        if not ms:
            return None
        counts: dict[str, list] = {}
        for c, pos in ms:
            counts.setdefault(c, [0, pos])[0] += 1
        return max(counts.items(), key=lambda kv: (kv[1][0], -kv[1][1]))[0]

    # ------------------------------------------------------------------ E2: class-name masking
    def _raw_pattern(self):
        """The lexicon variants as a regex on the *cleaned* title (accent-folded, lower-cased, same length), so
        a mention can be replaced in the text the encoder sees, keeping its punctuation and the rest of the title.
        Words of a variant may be separated by any non-alphanumeric run ('crab-cakes', 'mac & cheese')."""
        if getattr(self, "_raw", None) is None:
            def piece(variant):
                words, out = variant.split(), ""
                for i, w in enumerate(words):
                    if i:                                   # '&' needs no spaces around it ("mac&cheese")
                        out += r"[^a-z0-9]*" if "and" in (w, words[i - 1]) else r"[^a-z0-9]+"
                    out += r"(?:and|&)" if w == "and" else re.escape(w)
                return out
            alts = sorted(self.variant_to_class, key=len, reverse=True)
            self._raw = re.compile(r"(?<![a-z0-9])(?:" + "|".join(piece(a) for a in alts) + r")(?![a-z0-9])")
        return self._raw

    def mask(self, text, token: str = "[MASK]") -> str:
        """Every class-name mention (any class, all variants) replaced by `token` (DistilBERT's [MASK])."""
        t = clean_title(text)
        folded = "".join((unicodedata.normalize("NFKD", ch)[:1].lower()[:1] or " ") for ch in t)
        spans = [m.span() for m in self._raw_pattern().finditer(folded)]
        for a, b in reversed(spans):
            t = t[:a] + token + t[b:]
        return t
