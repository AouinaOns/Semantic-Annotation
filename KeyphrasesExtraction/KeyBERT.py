import os
import csv
import time
import pathlib
import torch
torch.set_num_threads(6)

from keybert import KeyBERT
from statistics import mean
from timeit import default_timer
from sentence_transformers import SentenceTransformer

def main():
    base_path = r'..\datasets'
    input_dir = os.path.join(base_path, 'docsutf8')
    output_dir = os.path.join(base_path, 'extracted/keybert')

    # Load model.
    model = KeyBERT(model = "camembert-base")

    # Set the current directory to the input dir
    os.chdir(os.path.join(os.getcwd(), input_dir))

    # Get all file names and their absolute paths.
    docnames = sorted(os.listdir())
    docpaths = list(map(os.path.abspath, docnames))

    # Create the keys directory, after the names and paths are loaded.
    pathlib.Path(output_dir).mkdir(parents = True, exist_ok = True)

    for i, (docname, docpath) in enumerate(zip(docnames, docpaths)):

        # keys shows up in docnames, erroneously.
        if docname == 'keys':
            continue
            
        print(f'Processing {i} out of {len(docnames)}...')

        # Save the output dir path
        output_dirpath = os.path.join(output_dir, docname.split('.')[0]+'.key')
        print(output_dirpath)

        with open(docpath, 'r', encoding = 'utf-8-sig', errors = 'ignore') as file, \
             open(output_dirpath, 'w', encoding = 'utf-8-sig', errors = 'ignore') as out:
            
            # Read the file and remove the newlines.
            text = file.read().replace('\n', ' ')

            # Extract the top 10 keyphrases.
            keyphrases_ranks = model.extract_keywords(text, keyphrase_ngram_range = (1, 4), 
                                       top_n = 10, diversity = 0.5, use_mmr = True)
            ranked_list = [k for k, r in keyphrases_ranks]
            
            # Write the keyphrases to file.
            keys = "\n".join(map(str, ranked_list) or '')
            out.write(keys)

        os.system('clear')
        s = 1
        print(f'Sleeping for {s} seconds...')
        time.sleep(s)


def find_nth_occurence(string, substring, start, end, n):
    """
    Function which finds the nth occurence of a 
    substring given a range of the original string.
    """

    i = string.find(substring, start, end)
    while i >= 0 and n > 1:
        i = string.find(substring, i + len(substring))
        n -= 1
    return i


def create_chunks(string, max_token_length, token_sep = ' '):
    """
    Function which creates chunks of max_token_length from a string,
    by using a token separator.
    """
    
    # Initialize the chunk range values.
    chunk_ranges = []
    chunk_start = 0
    chunk_end = 0

    # Find chunk ranges.
    while chunk_end < len(string):

        # Shift the chunk window to the next chunk.
        chunk_start = chunk_end

        # Find the next chunking position.
        next_sep_pos = find_nth_occurence(
            string, token_sep, chunk_start, len(string), 
            max_token_length
        )

        # If it is not found, the last chunk is smaller than the others.
        # Thus, we reached the end of the string.
        if next_sep_pos == -1:
            chunk_end = len(string)
        else:
            chunk_end = next_sep_pos

        chunk_ranges.append((chunk_start, chunk_end))
        
    # Construct the chunks of texts based on the previously calculated ranges.
    chunks = [string[i:j] for (i,j) in chunk_ranges]

    return chunks


def benchmark():
    text = """MODALITE D’HOSPITALISATION
HL Adressée par : le Docteur <nom />

MOTIF D’HOSPITALISATION
Patiente hospitalisée via le CMP à la demande du Docteur <nom /> pour
bilan et ajustement thérapeutique.

BIOGRAPHIE
Parents retraités vivant en <ville />.
Deuxième d’une fratrie de trois (deux frères).
Vit maritalement depuis plusieurs années.
Un fils <prenom />, âgée de 3 ans.
En invalidité depuis trois ans, madame <nom /> est diplômée en
comptabilité, métier qu’elle a exercé avant le début de sa maladie.

ANTECEDENTS MEDICOCHIRURGICAUX PERSONNELS
Endométriose cause de stérilité pendant 5 ans, traitée par voie
cœlioscopie.

ANTECEDENTS PSYCHIATRIQUES PERSONNELS
Début des troubles en janvier 2287 par un premier épisode psychotique traité
en ambulatoire par Haldol et Nozinan, avec persistance au décours d’un
important syndrome déficitaire.
Rechute psychotique en mars 2290 dans un contexte de rupture thérapeutique
et de post partum, compliqué d’un passage à l’acte autolytique.
Stabilisation sous Léponex à 425 mg par jour et suivi assuré au CMP par le.
Docteur <nom />

ANTECEDENTS FAMILIAUX
Frère psychotique suivi par le Docteur <nom /> au CMP de la Garenne
<ville />.

HISTOIRE DE LA MALADIE
Hospitalisation par le Docteur <nom /> de Madame <nom /> qui suite à
l’évolution déficitaire de sa maladie, s’est retrouvée en perte quasi-
totale d’autonomie et à la charge de son compagnon et de sa belle-mère.

EXAMEN A L’ENTREE
Patient incurique, angoissée.
Mâchonnement. Contact superficiel, affects abrasés.
Discours pauvre stéréotypé.
Pas d’activité délirante verbalisée.

EVOLUTION DANS LE SERVICE
Diminution des doses de Léponex jusqu’à 200 mg par jour ans le but
d’atténuer la sédation de la patiente et d’améliorer ses fonctions
cognitives.
Survenue d’un épisode infectieux broncho-pulmonaire prolongeant la durée de
l’hospitalisation.
Sous 200 mg de Léponex par jour, persistance du même tableau déficitaire
avec, cependant, en fin d’hospitalisation, net apaisement de l’angoisse.

PROJET DE SERVICE
Suivi ambulatoire par : le Docteur <nom />

Patiente confiée à sa famille (à sa mère pour un séjour de convalescence en
<ville />).
Hôpital de jour <prenom /> <rue />.
Accueil au CATTP.

EXAMENS COMPLEMENTAIRES
Bilan biologique standard : RAS.
Pharmacocinétique particulière du Léponex (à évaluer en ambulatoire : le
Docteur <nom /> doit nous en préciser les modalités).

AU TOTAL
Evolution déficitaire d’une schizophrénie désorganisée chez une patiente de
36 ans.

Une information concernant le motif d’hospitalisation, les indications
ainsi que les effets indésirables éventuels des examens complémentaires,
des traitements et de leurs implications a été donnée au patient ou à sa
famille.

TRAITEMENT DE SORTIE
CLOZAPINE 200 mg par jour.
ATARAX 100 mg 1- 0 – 1.
FORLAX sachet 1 par jour si constipation

DIAGNOSTIC
F 20.11
"""
    
    # Remove unnecessary whitespace from the text.
    text = ' '.join(text.split())

    # Calculate the token count.
    token_count = text.count(' ')

    # Load the model.


    sentence_model = SentenceTransformer("dangvantuan/sentence-camembert-large")
    model = KeyBERT(model=sentence_model)
    
    # Initialize the input token sizes and timings lists.
    mean_timings = []
    input_token_sizes = [250, 500, 1000, 2500, 5000, 6500, 8020]
    
    for n in input_token_sizes:

        # Select the first chunk of size n.
        selected_text = create_chunks(text, n)[0]

        # Amount of times to run the benchmarked approach.
        timings, repeat = [], 20
        for i in range(repeat):
            print(f'Text (#tokens): {n} -> Iteration ({i+1}/{repeat})')
            start_time = default_timer()
            model.extract_keywords(
                selected_text, keyphrase_ngram_range = (1, 4), 
                top_n = 10, diversity = 0.5, use_mmr = True
            )
            end_time = default_timer()
            timings.append(end_time - start_time)
            os.system('clear')

        mean_timings.append(mean(timings))

    # Save the input sizes and timings to a csv file.
    output_path = r'..\keybert_timings.csv'
    with open(output_path, 'w') as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerows(zip(input_token_sizes, mean_timings))

    return


if __name__ == '__main__': main()