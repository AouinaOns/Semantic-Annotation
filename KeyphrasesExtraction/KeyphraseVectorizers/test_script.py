import spacy
from keyphrase_vectorizers import KeyphraseCountVectorizer, KeyphraseTfidfVectorizer
import os
from keybert import KeyBERT
from flair.embeddings import TransformerDocumentEmbeddings
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge_sy"
# Date: 2020/2/21
import sys
import nltk
import time
import logging
import os
import re

def log_writing(logmsg,filename="./SIFRank/log/std.log"):
    # now we will Create and configure logger
    logging.basicConfig(filename=filename,
                        format='%(asctime)s %(message)s',
                        filemode='w')

    # Let us Create an object
    logger = logging.getLogger()

    # Now we are going to Set the threshold of logger to DEBUG
    logger.setLevel(logging.DEBUG)
    logger.info(f"{logmsg}")

def extract_sentences(text):
    # Replace line breaks with spaces to keep the sentences together
    text = text.replace('\n', ' ')
    # Split the text into sentences using periods as delimiters
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.)(\s|$)', text)

    # Remove empty strings and extra whitespace from each sentence
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    return sentences

def get_core_section(result):
    regex = r"^[A-Z][A-Z-:’'\n\s]+$\n"
    # retourne le contenu de la section
    matches = re.finditer(regex, result, re.MULTILINE)

    crh_tlt = ["Compte rendu d’hospitalisation"]
    crh_parts = []
    start_core = 0
    end_core = 0
    prev_start = 0
    for matchNum, match in enumerate(matches, start=1):
        end_core = match.start() - 1
        start = match.start()
        end = match.end()
        # print ("Match {matchNum} was found at {start}-{end}: {match}".format(matchNum = matchNum, start = match.start(), end = match.end(), match = match.group()))
        crh_tlt.append(match.group())
        # matches
        if matchNum == 1:
            text_core = result[start_core:start - 1]
            crh_parts.append(text_core)
            prev_start = end + 1
        else:

            text_core = result[prev_start:start - 1]
            start = start - 1
            crh_parts.append(text_core)


        prev_start = end

    text_core = result[prev_start:]
    crh_parts.append(text_core)
    lst_tuple = list(zip(crh_tlt, crh_parts))
    #return tuple list of pds section and sections name
    return lst_tuple




def read_text_file(file_path):
    with open(file_path, 'r',encoding="utf-8") as f:
        file=f.read()
    return file


def find_chunks_with_regex(text, chunk):
    matched_spans = []

    pattern = re.compile(chunk, re.IGNORECASE)
    for match in re.finditer(pattern, text):
            start, end = match.span()
            matched_text = text[start:end]
            matched_spans.append((matched_text, start, end))


    return matched_spans

def is_included(span1, span2):
    return span1 != span2 and span1[0] >= span2[0] and span1[1] <= span2[1]

def remove_included_strings(strings):
    strings = sorted(strings, key=lambda x: x[2] - x[1], reverse=True)  # Sort the strings by span length in descending order
    result = []
    for s in strings:
        if not any(is_included(s[1:], res[1:]) for res in result):
            result.append(s)
    return result

def generate_annotation_id(nb):
    # Generate a new annotation ID by counting existing annotations and adding 1
    return f"T{nb}"

def write_brat_annotation(filename, annotations):
    ann_type="annotationsToAdd"
    # Write the annotations to the .ann file
    ann_filename = filename.replace(".txt", ".ann")
    # Write the annotations to the .ann file
    with open(ann_filename, 'w+', encoding='utf-8') as ann_file:
        for i,(covered_text, start, end) in enumerate(annotations,1):
            ann_id = generate_annotation_id(i)
            ann_line = f"{ann_id}\t{ann_type} {start} {end}\t{covered_text}\n"
            ann_file.write(ann_line)


def keep_longest_overlapping_entities(entities):
    # Sort entities by their start position in ascending order
    entities.sort(key=lambda x: x[1])

    # List to store the filtered entities (longest non-overlapping entities)
    filtered_entities = []

    for entity in entities:
        # Check if the current entity overlaps with any entity in the filtered list
        overlap = False
        for filtered_entity in filtered_entities:
            if (entity[1] <= filtered_entity[2]) and (entity[2] >= filtered_entity[1]):
                overlap = True
                # Compare the lengths of the entities and keep the longest one
                if entity[2] - entity[1] > filtered_entity[2] - filtered_entity[1]:
                    filtered_entities.remove(filtered_entity)
                    filtered_entities.append(entity)
                break

        # If the current entity doesn't overlap with any entity in the filtered list, add it
        if not overlap:
            filtered_entities.append(entity)

    return filtered_entities



if __name__ == '__main__':

    path_output = "/Users/oaouina/Applications/onbasam/CRH_full_anonym"
    # now we will extract from each document its important np
    count=0
    doc_sections=[]
    list_core_sections = []
    for filename in os.listdir(path_output):

            pds_text = ""
            #for each pds we will extract np from each section
            # Check whether file is in text format or not
            keyphrases_l = []
            if filename.endswith(".txt"):


                #if count==1:
                    #break
                count += 1

                file_path = os.path.join(path_output, filename)
                # get the full pds
                fulltext=read_text_file(file_path)
                # extract the sections

                list_core_sections.extend(get_core_section(fulltext))
                print(count)
                if count>1000:
                    log_writing("end corpus constituition...")
                    break


    # we have all the docs
    doc_sections=[core for title,core in list_core_sections ]
    print(len(doc_sections))
    print(doc_sections[:10])
    # Init KeyBERT
    kw_model = KeyBERT(model=TransformerDocumentEmbeddings('camembert-base'))
    data_list=kw_model.extract_keywords(docs=doc_sections,
                                  vectorizer=KeyphraseCountVectorizer(spacy_pipeline='fr_dep_news_trf',
                                                                      pos_pattern='<ADJ.?>*<NOUN.*>+<ADJ.?>',
                                                                      stop_words='french'))

    # Extract the first elements of each tuple, skipping empty lists
    extracted_elements = []
    for sublist in data_list:
            for item in sublist:
                if item:  # Check if the tuple is not empty
                    extracted_elements.append(item[0])



            # Path to the text file
    file_path = "/Users/oaouina/Applications/pythonProject/KeyphraseVectorizers/annotation_full.txt"

    # Open the file in write mode and write the list to it
    f = open(file_path, 'w+')
    annotations_sorted = sorted(list(set(extracted_elements)), key=len)
    for item in annotations_sorted:
            f.write(item + '\n')

    f.close()


