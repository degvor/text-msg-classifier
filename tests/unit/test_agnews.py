"""Text normalisation. Pure -- needs no downloaded corpus."""

import pytest

from streamcls.corpus.agnews import Document, load_agnews, normalise_text


def test_strips_the_reuters_quote_link_artifact():
    """1.5 % of descriptions carry an HTML-escaped stock-quote link that occurs
    almost only in Business stories -- a free label the model must not get."""
    raw = (
        'Lowe\'s Cos. &lt;A HREF="http://www.investor.reuters.com/FullQuote.aspx?'
        'ticker=LOW.N target=/stocks/quickinfo/fullquote"&gt;LOW.N&lt;/A&gt; reported'
    )
    out = normalise_text(raw)
    assert "HREF" not in out
    assert "reuters.com" not in out
    assert "&lt;" not in out and "<" not in out
    assert "Lowe's Cos." in out and "reported" in out


def test_unescapes_double_encoded_entities():
    assert normalise_text("AT&amp;amp;T raised") == "AT&T raised"


def test_backslash_escapes_do_not_fuse_words():
    """AG News encodes line breaks as a literal backslash plus a letter."""
    assert normalise_text(r"dwindling\band of ultra-cynics") == "dwindling and of ultra-cynics"


def test_bare_urls_are_removed():
    out = normalise_text("Read at http://example.com/x?y=1 and www.foo.org/bar now")
    assert "http" not in out and "www." not in out
    assert out == "Read at and now"


def test_collapses_whitespace_and_strips():
    assert normalise_text("  a\t\tb \n c  ") == "a b c"


@pytest.mark.parametrize("raw", ["", "   ", "\\\\"])
def test_degenerate_input_yields_empty(raw):
    assert normalise_text(raw) == ""


def test_load_agnews_maps_labels_and_joins_fields(tmp_path):
    csv_path = tmp_path / "mini.csv"
    csv_path.write_text(
        '"1","World title","World body"\n'
        '"4","Tech title","Tech body"\n'
        '"3","Biz title","Biz body"\n',
        encoding="utf-8",
    )
    docs = list(load_agnews(csv_path))
    assert docs == [
        Document("World title World body", 0),
        Document("Tech title Tech body", 3),
        Document("Biz title Biz body", 2),
    ]


def test_load_agnews_skips_malformed_rows(tmp_path):
    csv_path = tmp_path / "mini.csv"
    csv_path.write_text('"1","only two fields"\n"2","ok","fine"\n', encoding="utf-8")
    assert [d.label for d in load_agnews(csv_path)] == [1]
