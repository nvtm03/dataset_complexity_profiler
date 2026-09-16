"""
HuggingFace Dataset Collector for Meta-Model Training.

This script fetches raw text datasets from the Hugging Face Hub, computes the
top-5 topological and geometric meta-features using the core profiler library,
and appends the results to a supplementary training CSV.

It explicitly filters out datasets that are already present in the primary
feature selection table to prevent data leakage and batch effects.

Usage:
    python research/collect_training_extra.py \
        --existing-csv research/feature_selection.csv \
        --output research/training_extra_collect.csv \
        --limit 1000
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

LOG = logging.getLogger("collect_training_extra")

GENERIC_TEXT_COLS = (
    "text",
    "sentence",
    "sentence1",
    "comment",
    "comment_text",
    "review",
    "content",
    "utterance",
    "Utterance",
    "claim",
    "string",
    "verse_text",
    "post",
    "question",
    "premise",
    "headline",
    "title",
    "statement",
    "passage",
)
GENERIC_LABEL_COLS = (
    "label",
    "labels",
    "gold_label",
    "sentiment",
    "emotion",
    "category",
    "topic",
    "intent",
    "class",
    "Label",
    "toxic",
    "toxicity",
    "hyperpartisan",
    "final_decision",
)

EXISTING_HF_KEYS = {
    "ag_news",
    "imdb",
    "sst2",
    "sst-2",
    "sst5",
    "sst-5",
    "rotten_tomatoes",
    "yelp_polarity",
    "yelp_review_full",
    "amazon_polarity",
    "amazon_reviews_multi_en",
    "amazon_counterfactual",
    "banking77",
    "clinc_oos",
    "clinc",
    "trec",
    "dbpedia",
    "dbpedia_14",
    "go_emotions",
    "goemotions",
    "tweet_eval/emoji",
    "tweet_eval/emotion",
    "tweet_eval/hate",
    "tweet_eval/offensive",
    "tweet_eval/sentiment",
    "glue/mnli",
    "glue/qqp",
    "glue/qnli",
    "glue/rte",
    "glue/mrpc",
    "glue/sst2",
    "glue/stsb",
    "paws",
    "paws-x/en",
    "snli",
    "xnli/en",
    "xnli/ru",
    "anli",
    "sick",
    "stsbenchmark",
    "stsb",
    "sts12",
    "sts13",
    "sts14",
    "sts15",
    "massive",
    "mtop_intent",
    "mtop",
    "setfit/20_newsgroups",
    "20_newsgroups",
    "bbc",
    "enron",
    "dair-ai/emotion",
    "emotion",
    "yahoo_answers_topics",
    "kinopoisk",
    "cedr",
    "georeview",
    "rugoemotions",
    "ru_goemotions",
    "rucola",
    "terra",
    "paraphraser",
    "rusentiment",
    "ruscibench",
    "sib-200",
    "sib200",
    "headline",
    "inappropriateness",
    "sensitive_topics",
    "tweetsentiment",
}

GATE_SKIPPED_NAMES = {
    "GLUE CoLA",
    "SuperGLUE BoolQ",
    "Civil Comments toxic",
    "GLUE WNLI",
    "SuperGLUE WiC",
    "SuperGLUE AX-b",
    "PubMedQA labeled",
    "Hate speech 18",
    "Toxic conversations",
    "TweetEval stance Hillary",
    "PAWS-X French",
    "PAWS-X German",
    "PAWS-X Spanish",
    "PAWS-X Japanese",
    "PAWS-X Chinese",
    "DaLAJ Swedish",
    "WikiQA",
}


def _src(
    name: str,
    hf: str,
    *,
    config: Optional[str] = None,
    split: str = "train",
    text: str = "text",
    label: str = "label",
    family: str = "single",
    lang: str = "en",
    **extra: Any,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "name": name,
        "hf": hf,
        "config": config,
        "split": split,
        "text": text,
        "label": label,
        "family": family,
        "lang": lang,
    }
    row.update(extra)
    return row


def _pair(
    name: str,
    hf: str,
    text: str,
    text_b: str,
    *,
    config: Optional[str] = None,
    lang: str = "en",
    **extra: Any,
) -> Dict[str, Any]:
    return _src(
        name,
        hf,
        config=config,
        text=text,
        text_b=text_b,
        family="pair",
        lang=lang,
        concat_single=True,
        **extra,
    )


CATALOG: List[Dict[str, Any]] = [
    _src(
        "Financial PhraseBank",
        "mteb/FinancialPhrasebankClassification",
        alt_hf=["financial_phrasebank", "takala/financial_phrasebank"],
        alt_text=["sentence"],
    ),
    _src("GLUE CoLA", "glue", config="cola", text="sentence"),
    _pair("SuperGLUE BoolQ", "super_glue", "question", "passage", config="boolq"),
    _src("LIAR fake-news", "liar", text="statement"),
    _src("TweetEval irony", "tweet_eval", config="irony"),
    _src("TweetEval stance climate", "tweet_eval", config="stance_climate"),
    _src(
        "Tweet topic single",
        "mteb/TweetTopicSingleClassification",
        alt_hf=["cardiffnlp/tweet_topic_single"],
        split="train",
        alt_splits=["train_2020", "train_2021"],
    ),
    _src("SetFit SUBJ", "SetFit/Subj"),
    _src("SetFit MPQA", "SetFit/MPQA", alt_hf=["SetFit/mpqa", "SetFit/mpqa_polarity"]),
    _src("PubMed RCT 20k", "armanc/pubmed-rct20k", alt_hf=["ml4pubmed/pubmed-rct20k"]),
    _src("ADE corpus classification", "ade_corpus_v2", config="Ade_corpus_v2_classification"),
    _src("Health fact", "health_fact", text="claim", alt_hf=["mteb/HealthfactClassification"]),
    _src("LexGLUE LEDGAR", "lex_glue", config="ledgar"),
    _src("LexGLUE SCOTUS", "lex_glue", config="scotus"),
    _src("LexGLUE Unfair ToS", "lex_glue", config="unfair_tos"),
    _src(
        "HateXplain",
        "hatexplain",
        text="post_tokens",
        join_tokens=True,
        label_from_annotators=True,
        alt_hf=["heegyu/hatexplain", "Hate-speech-CNERG/hatexplain"],
        alt_text=["text", "post"],
    ),
    _src(
        "Civil Comments toxic",
        "google/civil_comments",
        label="toxicity",
        binarize_at=0.5,
        alt_hf=["civil_comments"],
    ),
    _src("SILICONE MELD emotion", "silicone", config="meld_e", text="Utterance", label="Label"),
    _pair("RuSuperGLUE DaNetQA", "RussianNLP/russian_super_glue", "question", "passage", config="danetqa", lang="ru"),
    _pair("RuSuperGLUE RCB", "RussianNLP/russian_super_glue", "premise", "hypothesis", config="rcb", lang="ru"),
    _pair("RuSuperGLUE PARus", "RussianNLP/russian_super_glue", "premise", "choice1", config="parus", lang="ru"),
    _src(
        "RU toxic comments",
        "AlexSham/RUToxic",
        text="comment",
        label="toxic",
        lang="ru",
        alt_hf=["IlyaGusev/ru_toxic_comments", "textdetox/multilingual_toxicity_dataset", "s-nlp/ru_offensive"],
        alt_text=["text", "comment_text"],
        alt_label=["label", "toxic"],
    ),
    _pair("GLUE WNLI", "glue", "sentence1", "sentence2", config="wnli"),
    _pair("SuperGLUE CB", "super_glue", "premise", "hypothesis", config="cb"),
    _pair("SuperGLUE WiC", "super_glue", "sentence1", "sentence2", config="wic"),
    _src("SuperGLUE AX-b", "super_glue", config="axb", split="test", text="sentence1", text_b="sentence2", family="pair", concat_single=True),
    _src("SciCite", "allenai/scicite", text="string", alt_hf=["SetFit/scicite", "mteb/SciCiteClassification"]),
    _pair(
        "SciTail",
        "allenai/scitail",
        "sentence1",
        "sentence2",
        config="snli_format",
        label="gold_label",
        alt_hf=["SetFit/scitail"],
        alt_text=["premise", "text"],
        alt_label=["label"],
    ),
    _src("FEVER claim", "fever", config="v1.0", text="claim", alt_hf=["SetFit/fever", "mwong/fever-evidence-related"]),
    _pair("PubMedQA labeled", "pubmed_qa", "question", "context", config="pqa_labeled", label="final_decision"),
    _pair("MedNLI", "mednli", "sentence1", "sentence2", alt_hf=["bigbio/mednli"], label="gold_label"),
    _src("LexGLUE ECtHR A", "lex_glue", config="ecthr_a"),
    _src("LexGLUE EUR-LEX", "lex_glue", config="eurlex"),
    _src("MTOP Domain", "mteb/MTOPDomainClassification", config="en", alt_hf=["SetFit/mtop_domain"]),
    _src("SNIPS intents", "SetFit/snips", alt_hf=["benayas/snips"]),
    _src("ATIS intents", "SetFit/atis", alt_hf=["tuetschek/atis"]),
    _src("HWU64 intents", "mteb/HWU64Classification", alt_hf=["SetFit/hwu64", "SetFit/HWU64"]),
    _src("Hate speech 18", "SetFit/hate_speech18", alt_hf=["hate_speech18"]),
    _src("ETHOS binary", "SetFit/ethos-binary", alt_hf=["SetFit/ethos_binary", "ethos"]),
    _src("Toxic conversations", "SetFit/toxic_conversations", alt_hf=["mteb/ToxicConversationsClassification"]),
    _src("FRENK English hate", "mteb/FrenkEnClassification", alt_hf=["SetFit/frenk_en"]),
    _src(
        "Hyperpartisan news",
        "hyperpartisan_news_detection",
        config="byarticle",
        text="text",
        label="hyperpartisan",
        alt_splits=["train", "validation"],
        alt_hf=["SetFit/hyperpartisan_news", "mteb/HyperpartisanClassification"],
    ),
    _src("Reuters-21578", "yangwang825/reuters-21578", alt_hf=["SetFit/reuters21578", "reuters21578"]),
    _src(
        "OHSUMED",
        "ohsumed",
        text="title",
        text_b="abstract",
        concat_single=True,
        label="publication_type",
        alt_hf=["SetFit/ohsumed"],
        alt_label=["mesh_terms", "label"],
    ),
    _src("ArXiv classification", "mteb/ArxivClassification", alt_hf=["SetFit/arxiv_classification"]),
    _src("Patent classification", "mteb/PatentClassification", alt_hf=["SetFit/patent_classification"]),
    _src("App reviews", "mteb/AppReviewsClassification", alt_hf=["SetFit/app_reviews"]),
    _src("Poem sentiment", "google-research-datasets/poem_sentiment", text="verse_text", alt_hf=["SetFit/poem_sentiment"]),
    _src("Climate sentiment", "climatebert/climate_sentiment"),
    _src("Climate detection", "climatebert/climate_detection"),
    _src("Climate specificity", "climatebert/climate_specificity"),
    _src("SILICONE DailyDialog acts", "silicone", config="dyda_da", text="Utterance", label="Label"),
    _src("SILICONE IEMOCAP", "silicone", config="iemocap", text="Utterance", label="Label"),
    _src("SILICONE SEMAINE", "silicone", config="sem", text="Utterance", label="Label"),
    _src("TweetEval stance atheism", "tweet_eval", config="stance_atheism"),
    _src("TweetEval stance feminist", "tweet_eval", config="stance_feminist"),
    _src("TweetEval stance Hillary", "tweet_eval", config="stance_hillary"),
    _src("TweetEval stance abortion", "tweet_eval", config="stance_abortion"),
    _pair("PAWS-X French", "paws-x", "sentence1", "sentence2", config="fr", lang="fr"),
    _pair("PAWS-X German", "paws-x", "sentence1", "sentence2", config="de", lang="de"),
    _pair("PAWS-X Spanish", "paws-x", "sentence1", "sentence2", config="es", lang="es"),
    _pair("PAWS-X Japanese", "paws-x", "sentence1", "sentence2", config="ja", lang="ja"),
    _pair("PAWS-X Chinese", "paws-x", "sentence1", "sentence2", config="zh", lang="zh"),
    _pair("PAWS-X Korean", "paws-x", "sentence1", "sentence2", config="ko", lang="ko"),
    _pair("XNLI French", "xnli", "premise", "hypothesis", config="fr", lang="fr"),
    _pair("XNLI German", "xnli", "premise", "hypothesis", config="de", lang="de"),
    _pair("XNLI Spanish", "xnli", "premise", "hypothesis", config="es", lang="es"),
    _pair("XNLI Chinese", "xnli", "premise", "hypothesis", config="zh", lang="zh"),
    _pair("XNLI Arabic", "xnli", "premise", "hypothesis", config="ar", lang="ar"),
    _pair("XNLI Turkish", "xnli", "premise", "hypothesis", config="tr", lang="tr"),
    _pair("XNLI Hindi", "xnli", "premise", "hypothesis", config="hi", lang="hi"),
    _pair("XNLI Vietnamese", "xnli", "premise", "hypothesis", config="vi", lang="vi"),
    _src("Amazon Reviews DE", "SetFit/amazon_reviews_multi_de", lang="de"),
    _src("Amazon Reviews FR", "SetFit/amazon_reviews_multi_fr", lang="fr"),
    _src("Amazon Reviews ES", "SetFit/amazon_reviews_multi_es", lang="es"),
    _src("Amazon Reviews JA", "SetFit/amazon_reviews_multi_ja", lang="ja"),
    _src("Amazon Reviews ZH", "SetFit/amazon_reviews_multi_zh", lang="zh"),
    _src("Allociné reviews", "allocine", lang="fr", alt_hf=["SetFit/allocine"]),
    _src(
        "German sentiment",
        "tyqiangz/multilingual-sentiments",
        config="german",
        lang="de",
        alt_hf=["oliverguhr/german_sentiment"],
        alt_text=["text", "sentence"],
    ),
    _src("Tweet sentiment Arabic", "cardiffnlp/tweet_sentiment_multilingual", config="arabic", lang="ar"),
    _src("Tweet sentiment French", "cardiffnlp/tweet_sentiment_multilingual", config="french", lang="fr"),
    _src("Tweet sentiment German", "cardiffnlp/tweet_sentiment_multilingual", config="german", lang="de"),
    _src("Tweet sentiment Hindi", "cardiffnlp/tweet_sentiment_multilingual", config="hindi", lang="hi"),
    _src("Tweet sentiment Italian", "cardiffnlp/tweet_sentiment_multilingual", config="italian", lang="it"),
    _src("Tweet sentiment Portuguese", "cardiffnlp/tweet_sentiment_multilingual", config="portuguese", lang="pt"),
    _src("Tweet sentiment Spanish", "cardiffnlp/tweet_sentiment_multilingual", config="spanish", lang="es"),
    _src("Norwegian parliament", "mteb/NorwegianParliamentClassification", lang="no"),
    _src("DaLAJ Swedish", "mteb/DalajClassification", lang="sv"),
    _src("SweRec reviews", "mteb/SwedishSentimentClassification", lang="sv", alt_hf=["mteb/SweRecClassification"]),
    _src(
        "PolEmo2 IN",
        "clarin-pl/polemo2-official",
        config="in",
        lang="pl",
        alt_hf=["mteb/PolEmo2.0-INClassification"],
        alt_text=["text", "sentence"],
        alt_label=["label", "target"],
    ),
    _src("Greek legal code", "mteb/GreekLegalCodeClassification", lang="el"),
    _src("Czech product reviews", "mteb/CSFDCZMovieReviewSentimentClassification", lang="cs", alt_hf=["mteb/CzechProductReviewSentimentClassification"]),
    _src("Romanian reviews", "mteb/RomanianReviewsSentiment", lang="ro", alt_hf=["mteb/RomanianSentimentClassification"]),
    _src("Italian casehold", "mteb/ItaCaseholdClassification", lang="it"),
    _src(
        "Vietnamese student feedback",
        "uitnlp/vietnamese_students_feedback",
        lang="vi",
        alt_hf=["mteb/VietnameseStudentsFeedbackClassification"],
        alt_text=["sentence", "text"],
        alt_label=["sentiment", "label"],
    ),
    _pair("RuSuperGLUE RWSD", "RussianNLP/russian_super_glue", "text", "span1", config="rwsd", lang="ru", alt_text=["premise", "sentence1"]),
    _pair("RuSuperGLUE LiDiRus", "RussianNLP/russian_super_glue", "sentence1", "sentence2", config="lidirus", lang="ru", split="test"),
    _src("RuSuperGLUE RUSSE", "RussianNLP/russian_super_glue", config="russe", lang="ru", text="word", text_b="sentence1", family="pair", concat_single=True),
    _src(
        "RuReviews",
        "Tatyana/ru_sentiment_dataset",
        lang="ru",
        alt_hf=["sismetanin/rureviews"],
        alt_text=["text", "review", "comment"],
        alt_label=["sentiment", "label"],
    ),
    _src("TAPE ethics", "RussianNLP/tape", config="ethics.standard", lang="ru", alt_text=["text", "question"]),
    _src("TAPE sit ethics", "RussianNLP/tape", config="sit_ethics", lang="ru"),
    _src("WikiQA", "wiki_qa", text="question", text_b="answer", family="pair", concat_single=True, label="label"),
]


def _norm(text: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def _base_source_name(name: object) -> str:
    if _is_missing(name):
        return ""
    return str(name).split("[", 1)[0].strip().lower()


def load_existing_keys(existing_csv: Path) -> Set[str]:
    keys = {_norm(k) for k in EXISTING_HF_KEYS}
    if not existing_csv.is_file():
        LOG.warning("Existing CSV not found at %s — using built-in skip list only", existing_csv)
        return keys
    import pandas as pd

    df = pd.read_csv(existing_csv, usecols=["dataset_name"])
    for raw in df["dataset_name"].dropna().astype(str):
        base = _base_source_name(raw)
        keys.add(_norm(base))
        keys.add(_norm(raw))
    LOG.info("Loaded %s skip-keys from %s", len(keys), existing_csv)
    return keys


def catalog_keys(spec: Dict[str, Any]) -> List[str]:
    """Dedup keys. A bare HF id is not a skip-key when a config is set (else xnli drops all languages)."""
    parts = [_norm(spec["name"])]
    if spec.get("config"):
        parts.append(_norm(f"{spec['hf']}/{spec['config']}"))
    else:
        parts.append(_norm(spec["hf"]))
        for alt in spec.get("alt_hf") or []:
            parts.append(_norm(alt))
    return [p for p in parts if p]


def is_already_collected(spec: Dict[str, Any], existing: Set[str]) -> bool:
    return any(key in existing for key in catalog_keys(spec))


def _pick_split(ds_dict: Any, split: str, alt_splits: Sequence[str]) -> Any:
    if not hasattr(ds_dict, "keys"):
        return ds_dict
    for name in (split, *alt_splits):
        if name in ds_dict:
            return ds_dict[name]
    return ds_dict[next(iter(ds_dict.keys()))]


def _unique(seq: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(item for item in seq if item))


def load_hf_table(spec: Dict[str, Any], limit: int) -> Tuple[List[str], List[Any], str]:
    from datasets import load_dataset

    errors: List[str] = []
    hf_ids = [spec["hf"], *list(spec.get("alt_hf") or [])]
    splits = [spec.get("split") or "train", *list(spec.get("alt_splits") or [])]
    text_cols = _unique([spec["text"], *list(spec.get("alt_text") or []), *GENERIC_TEXT_COLS])
    label_cols = _unique([spec["label"], *list(spec.get("alt_label") or []), *GENERIC_LABEL_COLS])

    last_exc: Optional[BaseException] = None
    configs: List[Optional[str]] = []
    if spec.get("config"):
        configs.append(str(spec["config"]))
    configs.append(None)

    for hf in hf_ids:
        for cfg in configs:
            ds_args = (hf, cfg) if cfg else (hf,)
            try:
                raw = load_dataset(*ds_args)
            except MemoryError:
                raise
            except Exception as exc:
                if "trust_remote_code" not in str(exc).lower():
                    last_exc = exc
                    errors.append(f"{hf}: {exc}")
                    continue
                LOG.warning(
                    "Dataset %s requires remote code; retrying with trust_remote_code=True",
                    hf,
                )
                try:
                    raw = load_dataset(*ds_args, trust_remote_code=True)
                except MemoryError:
                    raise
                except Exception as inner_exc:
                    last_exc = inner_exc
                    errors.append(f"{hf} (remote_code): {inner_exc}")
                    continue
            try:
                ds = _pick_split(raw, splits[0], splits[1:])
                cols = set(ds.column_names)
                text_col = next((c for c in text_cols if c in cols), None)
                label_col = next((c for c in label_cols if c in cols), None)
                if text_col is None or label_col is None:
                    errors.append(f"{hf}: columns {sorted(cols)} missing {text_cols[:6]}/{label_cols[:6]}")
                    continue
                if len(ds) > limit:
                    ds = ds.shuffle(seed=42)
                texts: List[str] = []
                labels: List[Any] = []
                text_b_col = spec.get("text_b") if spec.get("text_b") in cols else None
                for row in ds:
                    left = row[text_col]
                    if spec.get("join_tokens") and isinstance(left, (list, tuple)):
                        left = " ".join(str(t) for t in left)
                    if _is_missing(left) or not str(left).strip():
                        continue
                    if spec.get("label_from_annotators") and isinstance(row.get(label_col), dict):
                        lab = row[label_col].get("majority")
                        if lab is None:
                            lab = row[label_col]
                    else:
                        lab = row[label_col]
                    if isinstance(lab, (list, tuple)):
                        if not lab:
                            continue
                        lab = lab[0]
                    if spec.get("binarize_at") is not None:
                        try:
                            lab = int(float(lab) >= float(spec["binarize_at"]))
                        except (TypeError, ValueError):
                            continue
                    if _is_missing(lab):
                        continue
                    if text_b_col and spec.get("concat_single"):
                        right = row[text_b_col]
                        if isinstance(right, (list, tuple)):
                            right = " ".join(str(t) for t in right)
                        if _is_missing(right) or not str(right).strip():
                            continue
                        left = f"{left} [SEP] {right}"
                    texts.append(str(left))
                    labels.append(lab)
                    if len(texts) >= limit:
                        break
                if len(texts) < 80:
                    errors.append(f"{hf}: only {len(texts)} usable rows")
                    continue
                family = "single" if spec.get("concat_single") else str(spec.get("family") or "single")
                return texts, labels, family
            except MemoryError:
                raise
            except Exception as exc:
                last_exc = exc
                errors.append(f"{hf}: {exc}")
                continue
    raise RuntimeError(" | ".join(errors) or str(last_exc))


def profile_one(
    texts: Sequence[str],
    labels: Sequence[Any],
    spec: Dict[str, Any],
    family: str,
    embedder: str,
    batch_size: int,
    quality_threshold: float,
) -> Optional[Dict[str, Any]]:
    import numpy as np

    from dataset_complexity_profiler import DatasetProfiler
    from dataset_complexity_profiler.empirical import estimate_intrinsic_dim
    from dataset_complexity_profiler.features import build_meta_feature_vector
    from dataset_complexity_profiler.meta_artifact import (
        INTERPRETABLE_FEATURE_NAMES,
        separability_row_values,
    )
    from dataset_complexity_profiler.probes import linear_separability_status

    profiler = DatasetProfiler(auto_load_meta_model=False)
    LOG.info("Embedding %s (%s texts)", spec["name"], len(texts))
    X = profiler.embed_texts(
        list(texts),
        embedder_name=embedder,
        batch_size=batch_size,
        show_progress=True,
    )
    y = np.asarray(labels)
    named = build_meta_feature_vector(X, y, task_family=family)
    sep = linear_separability_status(float(named["baseline_quality"]), y, task_family=family)
    if not sep.get("passed"):
        LOG.warning("SKIP %s: %s", spec["name"], sep.get("message"))
        return None
    dim_info = estimate_intrinsic_dim(
        X,
        y,
        quality_threshold=quality_threshold,
        min_dim_bound=2,
        task_family=family,
        stop_at_threshold=True,
    )
    row = {
        "dataset_name": spec["name"],
        "recommended_dim": int(dim_info["recommended_dim"]),
        "task_family": family,
        "true_intrinsic_dim": "",
        "is_synthetic": 0,
        "language": spec.get("lang") or "en",
    }
    for feat in INTERPRETABLE_FEATURE_NAMES:
        row[feat] = float(named[feat])
    metric, value, threshold = separability_row_values(sep)
    row["linear_sep_metric"] = metric
    row["linear_sep_value"] = value
    row["linear_sep_threshold"] = threshold
    LOG.info(
        "OK %s dim=%s kappa/spearman=%.3f class_count=%s",
        spec["name"],
        row["recommended_dim"],
        float(value or 0),
        row["class_count"],
    )
    return row


def _default_output() -> Path:
    return Path("research/training_extra_collect.csv")

def _default_existing() -> Path:
    return Path("research/feature_selection.csv")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--existing-csv",
        type=Path,
        default=_default_existing(),
        help="feature-selection table used to skip already collected sources",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_default_output(),
        help="scratch CSV from this run (gitignored); curated extra is research/training_extra.csv",
    )
    parser.add_argument("--limit", type=int, default=1000, help="Max texts per source")
    parser.add_argument("--embedder", default="paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--quality-threshold", type=float, default=0.97)
    parser.add_argument("--only", default="", help="Comma-separated substrings of source names to keep")
    parser.add_argument("--max-sources", type=int, default=0, help="Stop after N new sources this run (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="Print planned sources and exit")
    parser.add_argument("--no-resume", action="store_true", help="Overwrite --output instead of appending")
    return parser.parse_args(argv)


def planned_catalog(existing: Set[str], only: str) -> List[Dict[str, Any]]:
    needles = [n.strip().lower() for n in only.split(",") if n.strip()]
    planned: List[Dict[str, Any]] = []
    for spec in CATALOG:
        if needles and not any(n in spec["name"].lower() or n in spec["hf"].lower() for n in needles):
            continue
        if is_already_collected(spec, existing):
            LOG.info("Already in selection CSV, skip: %s (%s)", spec["name"], spec["hf"])
            continue
        if spec["name"] in GATE_SKIPPED_NAMES:
            LOG.info("Previously failed linear-sep gate, skip: %s", spec["name"])
            continue
        planned.append(spec)
    return planned


def _written_names(output: Path) -> Set[str]:
    if not output.is_file() or output.stat().st_size == 0:
        return set()
    import pandas as pd

    try:
        df = pd.read_csv(output, usecols=["dataset_name"])
    except (ValueError, pd.errors.EmptyDataError):
        return set()
    return set(df["dataset_name"].dropna().astype(str))


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    existing = load_existing_keys(args.existing_csv)
    planned = planned_catalog(existing, args.only)
    resume = not args.no_resume
    done = _written_names(args.output) if resume else set()
    if done:
        planned = [spec for spec in planned if spec["name"] not in done]
        LOG.info("Resume: %s names already in %s", len(done), args.output)
    if args.max_sources and args.max_sources > 0:
        planned = planned[: args.max_sources]
    LOG.info("Will collect %s training-extra sources (catalog had %s)", len(planned), len(CATALOG))
    for spec in planned:
        LOG.info("  - %s  [%s%s]", spec["name"], spec["hf"], f"/{spec['config']}" if spec.get("config") else "")
    if args.dry_run:
        return 0
    if not planned:
        LOG.warning("Nothing to collect.")
        return 0

    from dataset_complexity_profiler.meta_artifact import META_CSV_COLUMNS

    fieldnames = [c for c in META_CSV_COLUMNS if c not in {"sample_count", "original_dim", "minority_fraction"}]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_fail = 0
    write_header = args.no_resume or not args.output.is_file() or args.output.stat().st_size == 0
    mode = "w" if args.no_resume or write_header else "a"
    with args.output.open(mode, encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            fh.flush()
        for spec in planned:
            try:
                texts, labels, family = load_hf_table(spec, args.limit)
                row = profile_one(
                    texts,
                    labels,
                    spec,
                    family,
                    args.embedder,
                    args.batch_size,
                    args.quality_threshold,
                )
            except MemoryError:
                raise
            except Exception:
                LOG.exception("FAILED %s", spec["name"])
                n_fail += 1
                continue
            if row is None:
                n_fail += 1
                continue
            writer.writerow(row)
            fh.flush()
            n_ok += 1
    LOG.info("Wrote %s rows to %s (%s failed/skipped)", n_ok, args.output, n_fail)
    return 0 if (n_ok or done) else 1


if __name__ == "__main__":
    sys.exit(main())
