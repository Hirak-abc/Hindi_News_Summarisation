from textrank.summarizer import split_sentences, summarize_text


def test_split_sentences_preserves_hindi_danda():
    # Hindi danda should remain at the end of each split sentence.
    article = "भारत ने जीत दर्ज की। खिलाड़ी खुश हैं।"
    assert split_sentences(article) == [
        "भारत ने जीत दर्ज की।",
        "खिलाड़ी खुश हैं।",
    ]


def test_summarize_empty_article():
    # An empty source article should produce an empty summary.
    assert summarize_text("") == ""


def test_summarize_single_sentence_respects_word_limit():
    # Single-sentence input should be truncated to the configured word cap.
    article = " ".join(["शब्द"] * 20)
    summary = summarize_text(article, max_words=5)
    assert summary == " ".join(["शब्द"] * 5)
