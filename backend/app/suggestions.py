from __future__ import annotations

from collections import Counter

from .search import VideoRecord, tokenize


def summarize_patterns(records: list[VideoRecord]) -> list[str]:
    if not records:
        return ["No close matches were found, so suggestions are based on general metadata quality heuristics."]

    title_lengths = [len(record.title) for record in records if record.title]
    avg_title_length = round(sum(title_lengths) / len(title_lengths)) if title_lengths else 0

    tag_counter: Counter[str] = Counter()
    token_counter: Counter[str] = Counter()
    categories = Counter(record.category_id for record in records if record.category_id)

    for record in records:
        tag_counter.update(tag.lower() for tag in record.tags[:10])
        token_counter.update(tokenize(record.title))

    summary = []
    if categories:
        category, count = categories.most_common(1)[0]
        summary.append(f"Most similar matches cluster around category {category} ({count}/{len(records)} results).")
    if avg_title_length:
        summary.append(f"Similar trending titles average about {avg_title_length} characters, which is a good target range.")
    if tag_counter:
        common_tags = ", ".join(tag for tag, _ in tag_counter.most_common(5))
        summary.append(f"Frequent tags in matching videos: {common_tags}.")
    if token_counter:
        common_title_terms = ", ".join(term for term, _ in token_counter.most_common(5))
        summary.append(f"Frequent title terms in matching videos: {common_title_terms}.")
    return summary


def build_suggestions(title: str, description: str, content: str, tags: list[str], records: list[VideoRecord]) -> list[dict]:
    title_tokens = tokenize(title)
    metadata_body = " ".join(part for part in [description, content] if part.strip())
    desc_tokens = tokenize(metadata_body)
    reference_title_tokens = Counter()
    reference_tags = Counter()

    for record in records:
        reference_title_tokens.update(tokenize(record.title))
        reference_tags.update(tag.lower() for tag in record.tags)

    suggestions: list[dict] = []

    if len(title.strip()) < 25:
        suggestions.append(
            {
                "type": "title",
                "message": "Your title is quite short. Similar trending videos usually add one stronger hook, outcome, or named entity.",
            }
        )
    if not any(char.isdigit() for char in title):
        suggestions.append(
            {
                "type": "title",
                "message": "Consider testing a title variant with a number, episode marker, ranking, or concrete payoff if it fits the content.",
            }
        )
    if len(metadata_body.strip()) < 80:
        suggestions.append(
            {
                "type": "description",
                "message": "Your metadata body is sparse. Matching videos often add a short context block plus keywords that reinforce discoverability.",
            }
        )
    if len(desc_tokens) < 20:
        suggestions.append(
            {
                "type": "positioning",
                "message": "You have not given the system much topic context yet. A 2-3 sentence content summary will usually improve match quality and downstream suggestions.",
            }
        )
    if len(tags) < 4:
        suggestions.append(
            {
                "type": "tags",
                "message": "Your tag set is thin. Similar trending videos usually repeat the main topic with alternate names, formats, and audience terms.",
            }
        )
    if reference_title_tokens:
        overlap = [token for token in title_tokens if token in reference_title_tokens]
        if len(overlap) < 2:
            candidate_terms = [term for term, _ in reference_title_tokens.most_common(3)]
            suggestions.append(
                {
                    "type": "positioning",
                    "message": f"Your wording is not very close to top matches. Consider naturally weaving in terms like: {', '.join(candidate_terms)}.",
                }
            )
    if reference_tags:
        existing_tags = {tag.lower() for tag in tags}
        candidate_tags = [tag for tag, _ in reference_tags.most_common(5) if tag not in existing_tags]
        if candidate_tags:
            suggestions.append(
                {
                    "type": "tags",
                    "message": f"Add a few high-signal tags from similar videos, such as: {', '.join(candidate_tags[:3])}.",
                }
            )

    return suggestions[:5]


def build_draft_titles(title: str, records: list[VideoRecord]) -> list[str]:
    if not records:
        return [title]

    terms = []
    for record in records:
        for token, _ in Counter(tokenize(record.title)).most_common(4):
            if token not in terms:
                terms.append(token)
    terms = terms[:4]

    base = title.strip()
    drafts = [
        base,
        f"{base} | What Similar Trending Videos Are Doing Right",
        f"{base} - {', '.join(terms[:2]).title()}" if terms else base,
    ]

    deduped = []
    for draft in drafts:
        cleaned = " ".join(draft.split())
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return deduped[:3]


def build_draft_tags(tags: list[str], records: list[VideoRecord]) -> list[str]:
    seen = {tag.lower() for tag in tags}
    merged = list(tags)
    counts = Counter()
    for record in records:
        counts.update(tag.lower() for tag in record.tags)

    for tag, _ in counts.most_common(8):
        if tag not in seen:
            merged.append(tag)
            seen.add(tag)
        if len(merged) >= 10:
            break
    return merged
