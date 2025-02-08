from typing import List

import flair
import spacy
from flair.models import SequenceTagger
from flair.splitter import SegtokSentenceSplitter
from keybert import KeyBERT

import tests.utils as utils
from keyphrase_vectorizers import KeyphraseCountVectorizer, KeyphraseTfidfVectorizer

english_docs = utils.get_english_test_docs()
german_docs = utils.get_german_test_docs()
french_docs = utils.get_french_docs()


def test_default_count_vectorizer():
    sorted_english_test_keyphrases = utils.get_english_test_keyphrases()
    sorted_count_matrix = utils.get_sorted_english_count_matrix()

    vectorizer = KeyphraseCountVectorizer()
    vectorizer.fit(english_docs)
    keyphrases = vectorizer.get_feature_names_out()
    document_keyphrase_matrix = vectorizer.transform(english_docs).toarray()

    assert [sorted(count_list) for count_list in
            KeyphraseCountVectorizer().fit_transform(english_docs).toarray()] == sorted_count_matrix
    assert [sorted(count_list) for count_list in document_keyphrase_matrix] == sorted_count_matrix
    assert sorted(keyphrases) == sorted_english_test_keyphrases


def test_spacy_language_argument():
    sorted_english_test_keyphrases = utils.get_english_test_keyphrases()
    sorted_count_matrix = utils.get_sorted_english_count_matrix()

    nlp = spacy.load("en_core_web_sm")

    vectorizer = KeyphraseCountVectorizer(spacy_pipeline=nlp)
    vectorizer.fit(english_docs)
    keyphrases = vectorizer.get_feature_names_out()
    document_keyphrase_matrix = vectorizer.transform(english_docs).toarray()

    assert [sorted(count_list) for count_list in
            KeyphraseCountVectorizer().fit_transform(english_docs).toarray()] == sorted_count_matrix
    assert [sorted(count_list) for count_list in document_keyphrase_matrix] == sorted_count_matrix
    assert sorted(keyphrases) == sorted_english_test_keyphrases


def test_german_count_vectorizer():
    sorted_german_test_keyphrases = utils.get_german_test_keyphrases()

    vectorizer = KeyphraseCountVectorizer(spacy_pipeline='de_core_news_sm', pos_pattern='<ADJ.*>*<N.*>+',
                                          stop_words='german')
    keyphrases = vectorizer.fit(german_docs).get_feature_names_out()
    assert sorted(keyphrases) == sorted_german_test_keyphrases


def test_default_tfidf_vectorizer():
    sorted_english_test_keyphrases = utils.get_english_test_keyphrases()
    sorted_english_tfidf_matrix = utils.get_sorted_english_tfidf_matrix()

    vectorizer = KeyphraseTfidfVectorizer()
    vectorizer.fit(english_docs)
    keyphrases = vectorizer.get_feature_names_out()
    document_keyphrase_matrix = vectorizer.transform(english_docs).toarray()
    document_keyphrase_matrix = [[round(element, 10) for element in tfidf_list] for tfidf_list in
                                 document_keyphrase_matrix]

    assert [sorted(tfidf_list) for tfidf_list in document_keyphrase_matrix] == sorted_english_tfidf_matrix
    assert sorted(keyphrases) == sorted_english_test_keyphrases


def test_keybert_integration():
    english_keybert_keyphrases = utils.get_english_keybert_keyphrases()
    kw_model = KeyBERT(model="all-MiniLM-L6-v2")
    keyphrases = kw_model.extract_keywords(docs=english_docs, vectorizer=KeyphraseCountVectorizer())
    keyphrases = [[element[0] for element in keyphrases_list] for keyphrases_list in keyphrases]

    assert keyphrases == english_keybert_keyphrases


def test_french_trf_spacy_pipeline():
    sorted_french_test_keyphrases = utils.get_french_test_keyphrases()
    sorted_french_count_matrix = utils.get_sorted_french_count_matrix()

    vectorizer = KeyphraseCountVectorizer(spacy_pipeline='fr_dep_news_trf')
    vectorizer.fit(french_docs)
    keyphrases = vectorizer.get_feature_names_out()
    document_keyphrase_matrix = vectorizer.transform(french_docs).toarray()

    assert [sorted(count_list) for count_list in
            KeyphraseCountVectorizer(spacy_pipeline='fr_dep_news_trf').fit_transform(
                french_docs).toarray()] == sorted_french_count_matrix
    assert [sorted(count_list) for count_list in document_keyphrase_matrix] == sorted_french_count_matrix
    assert sorted(keyphrases) == sorted_french_test_keyphrases
    print(sorted_french_test_keyphrases)

def test_custom_tagger():
    sorted_english_test_keyphrases = utils.get_sorted_english_keyphrases_custom_flair_tagger()

    tagger = SequenceTagger.load('pos')
    splitter = SegtokSentenceSplitter()

    # define custom pos tagger function using flair
    def custom_pos_tagger(raw_documents: List[str], tagger: flair.models.SequenceTagger = tagger,
                          splitter: flair.splitter.SegtokSentenceSplitter = splitter) -> List[tuple]:
        """
        Important:

        The mandatory 'raw_documents' parameter can NOT be named differently and has to expect a list of strings.
        Furthermore the function has to return a list of (word token, POS-tag) tuples.
        """
        # split texts into sentences
        sentences = []
        for doc in raw_documents:
            sentences.extend(splitter.split(doc))

        # predict POS tags
        tagger.predict(sentences)

        # iterate through sentences to get word tokens and predicted POS-tags
        pos_tags = []
        words = []
        for sentence in sentences:
            pos_tags.extend([label.value for label in sentence.get_labels('pos')])
            words.extend([word.text for word in sentence])
        print(list(zip(words, pos_tags)))
        return list(zip(words, pos_tags))

    vectorizer = KeyphraseCountVectorizer(custom_pos_tagger=custom_pos_tagger)
    vectorizer.fit(english_docs)
    keyphrases = vectorizer.get_feature_names_out()
    print(sorted_english_test_keyphrases)
    assert sorted(keyphrases) == sorted_english_test_keyphrases

if __name__ == "__main__":
    docs = ["""1ère hospitalisation en 2741, à l’âge de 24ans à l’hôpital <ville />, pour syndrome délirant dans un contexte de consommation de cannabis. Hospitalisation d’un mois. Rupture de soins à la sortie.

    2ème hospitalisation en novembre 2742 en <ville />, puis hôpital de <ville /> pendant 1 mois pour recrudescence délirante (persécution, messianique). RISPERDAL prescrit, efficace mais rechute lors de l’arrêt du traitement suivi d’une hospitalisation à <ville /> pendant plusieurs mois.

    Pendant 1 an : pas d’hospitalisation, a débuté un DAEU : Diplôme d’accès aux études universitaires, faisait du sport.

    Dernière hospitalisation au <hôpital /> au secteur 16 en janvier 2744 jusqu’à mai 2744 pour syndrome délirant sur rupture de traitement. Sortie sous HALDOL 10mg.

    Prise en charge à l’HDJ : accède au foyer <ville /> Capitant fin 2744.

    Remplacement de l’HALDOL par l’OLANZAPINE 10mg dans l’objectif de la mise en place d’une forme retard, à l’été 2745.

    Intégration au groupe NEAR fin 2745.""",

            """Intègre en mars 2746 le CRP <prenom /> <nom /> pour une formation en informatique.

    Projet de déménagement lancé en mars 2747 de Capitant à un appartement thérapeutique en vue ensuite d’un HLM.

    Antécédents somatiques personnels : Descente de testicules opératoire vers 4-5 ans unilatérale.

    Antécédents addictologiques :

    Trouble de l’usage du cannabis.

    Tabagisme.

    Antécédents familiaux :

    Grand frère, mère et demi-frère ont un suivi en psychiatrie, ont été hospitalisés.

    Son demi-frère a été hospitalisé pour bouffées délirantes et angoisses.

    Allergies : Pollens

    Eléments biographiques et mode de vie :

    Né au <pays />, à Taroudant.

    En <pays /> depuis qu’il est nourrisson. Sa mère a décidé de venir en <pays /> après le divorce.

    2 frères aînés. 3 demi-frères et 1 demi-sœur côté maternel. 3 demi-sœurs côté paternel Ses frères sont arrivés à l’âge de 12 ans (avant cela père avait la garde) L’entente est bonne avec ses frères et sœurs.

    Placé, intègre les orphelins d’<ville />. Son beau-père aurait un trouble de l’usage de l’alcool.

    Mère gardienne d’immeuble, habite à <ville />. Frères habitent à <ville />.

    Scolarité : en échec scolaire depuis CM2. Redoublement en classe de 6ème et 1ère. S’est arrêté en classe de 1ère STG (technologique).

    A ensuite fait des animations à <ville /> dans des écoles primaires/maternelles pendant 1 an mais arrêt du fait d’un revenu trop bas

    De 19 à 21 ans : périodes d’errances après avoir quitté le domicile du fait de violences du beau-père, il loge dans une chambre de bonne puis 115. Un ami l’héberge quelque temps, puis il intègre l’armée à 21 ans. Il fait désertion car « ne voulait pas aller en <pays /> ». Est allé vivre à <ville />. A travaillé chez Nestlé, conditionnement pendant 7 mois. Contrat interrompu car n’a pas réussi à tenir les horaires : période où fumait beaucoup de cannabis."""]
    nlp = spacy.load("fr_dep_news_trf")
    vectorizer1 = KeyphraseCountVectorizer(spacy_pipeline=nlp)
    print(vectorizer1.fit(docs))


