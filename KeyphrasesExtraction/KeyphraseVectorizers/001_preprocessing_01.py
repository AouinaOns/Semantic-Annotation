import spacy
from spaczz.matcher import SimilarityMatcher
from spacy.tokens import Span
from spacy.matcher import PhraseMatcher
from typing import Iterable, List, Set
from keybert import KeyBERT
from flair.embeddings import TransformerDocumentEmbeddings
from keyphrase_vectorizers import KeyphraseCountVectorizer, KeyphraseTfidfVectorizer
# Init vectorizer for the french language
#all-mpnet-base-v2 
kw_model = KeyBERT(model=TransformerDocumentEmbeddings('dangvantuan/sentence-camembert-large'))


class Span:
    def __init__(self, entity_type, start, end, text):
        self.entity_type = entity_type
        self.start = start
        self.end = end
        self.text = text

def filter_spans(spans: Iterable[Span]) -> List[Span]:
    """Filter a sequence of spans and remove duplicates or overlaps. Useful for
    creating named entities (where one token can only be part of one entity) or
    when merging spans with `Retokenizer.merge`. When spans overlap, the (first)
    longest span is preferred over shorter spans.

    spans (Iterable[Span]): The spans to filter.
    RETURNS (List[Span]): The filtered spans.
    """
    get_sort_key = lambda span: (span.end - span.start, -span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens: Set[int] = set()
    for span in sorted_spans:
        # Check for end - 1 here because boundaries are inclusive
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
            seen_tokens.update(range(span.start, span.end))
    result = sorted(result, key=lambda span: span.start)
    return result


def convert_to_brat_ann(spans):
    lines = []
    for idx, span in enumerate(spans, start=1):
        entity_type = span.entity_type
        start = span.start
        end = span.end  # End position is exclusive in spaCy
        text = span.text
        line = f'T{idx}\t{entity_type} {start} {end}\t{text}\n'
        lines.append(line)
    return ''.join(lines)

file_path=".../KeyphraseVectorizers/annotation_art1.txt"
# Read the lines from the text file and put them into a list
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Remove leading and trailing whitespace from each line and create the list
entities_list = [line.strip() for line in lines]

nlp = spacy.load("fr_core_news_lg")

def match_entities(text):
    doc = nlp(text)

    matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")

    # Iterate over the patterns and add them to the Matcher
    for pattern in entities_list:
        matcher.add("ENTITY", [nlp(pattern.lower())])


    matches = matcher(doc)
    #get the spans from a char in spacy
    spans = []
    for match_id, start, end in matches:
        span = doc[start: end]
        print(span.text, span.start_char, span.end_char)
        spans.append(("ENTITY", span.start_char, span.end_char, span.text))



vectorizer1 = KeyphraseCountVectorizer(spacy_pipeline='fr_dep_news_trf', pos_pattern='<ADJ.*>*<NOUN.*>+<ADP.?><NOUN.*>+<ADJ.*>*')
keyphrases_nouns = vectorizer1.get_feature_names_out()
print(keyphrases_nouns)

vectorizer1 = KeyphraseCountVectorizer(spacy_pipeline='fr_dep_news_trf', pos_pattern='<ADJ.*>*<NOUN.*>+<ADP.?><NOUN.*>+<ADJ.*>*')
keyphrases = vectorizer1.get_feature_names_out()


