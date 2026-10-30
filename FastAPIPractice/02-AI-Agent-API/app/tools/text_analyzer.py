import re
from collections import Counter
from typing import Dict, List

COMMON_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it",
    "its", "itself", "let's", "me", "more", "most", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "with", "won't", "would", "you", "your", "yours", "yourself",
}


def analyze_text(text: str) -> str:
    """
    Analyze text to compute statistics including word count, character count, sentence count, estimated reading time, and key frequent terms.

    Args:
        text: The text content to analyze.

    Returns:
        A structured string summarizing the text metrics.
    """
    if not text or not text.strip():
        return "The provided text is empty."

    raw_text = text.strip()
    char_count_total = len(raw_text)
    char_count_no_spaces = len(re.sub(r"\s+", "", raw_text))

    # Words
    words = re.findall(r"\b[a-zA-Z0-9_\'-]+\b", raw_text)
    word_count = len(words)

    # Sentences
    sentences = re.split(r"[.!?]+", raw_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = max(1, len(sentences))

    # Average word length
    avg_word_length = round(sum(len(w) for w in words) / max(1, word_count), 2)

    # Reading time calculation (average 200 words per minute)
    reading_seconds = round((word_count / 200) * 60)
    reading_time_str = f"{reading_seconds // 60}m {reading_seconds % 60}s" if reading_seconds >= 60 else f"{reading_seconds}s"

    # Top keywords (excluding common stop words and single character tokens)
    clean_tokens = [w.lower() for w in words if w.lower() not in COMMON_STOPWORDS and len(w) > 2]
    top_words = Counter(clean_tokens).most_common(5)
    top_words_formatted = ", ".join([f"'{word}' ({count}x)" for word, count in top_words]) or "None"

    return (
        f"Text Analysis Results:\n"
        f"- Total Words: {word_count}\n"
        f"- Total Characters (with spaces): {char_count_total}\n"
        f"- Characters (without spaces): {char_count_no_spaces}\n"
        f"- Sentences: {sentence_count}\n"
        f"- Average Word Length: {avg_word_length} chars\n"
        f"- Estimated Reading Time: ~{reading_time_str}\n"
        f"- Top Keywords: {top_words_formatted}"
    )
