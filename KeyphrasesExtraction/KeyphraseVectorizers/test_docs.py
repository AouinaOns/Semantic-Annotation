#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge_sy"
# Date: 2020/2/21
import sys
import nltk
from embeddings import sent_emb_sif, word_emb_elmo
from model.method import SIFRank, SIFRank_plus
from stanfordcorenlp import StanfordCoreNLP
import time
import logging
import os
import re

def log_writing(logmsg,filename="/Users/oaouina/Applications/pythonProject/SIFRank/log/std.log"):
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




#download from https://allennlp.org/elmo
options_file = "/Users/oaouina/Applications/pythonProject/SIFRank/auxiliary_data/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "/Users/oaouina/Applications/pythonProject/SIFRank/auxiliary_data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
#options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
#weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
porter = nltk.PorterStemmer()
ELMO = word_emb_elmo.WordEmbeddings(options_file, weight_file)
SIF = sent_emb_sif.SentEmbeddings(ELMO, lamda=1.0)
en_model = StanfordCoreNLP(r'/Users/oaouina/Applications/pythonProject/SIFRank/stanford-corenlp-4.0.0',quiet=True,lang="fr")#download from https://stanfordnlp.github.io/CoreNLP/
elmo_layers_weight = [0.0, 1.0, 0.0]


def read_text_file(file_path):
    with open(file_path, 'r',encoding="utf-8") as f:
        file=f.read()
    return file

def extract_SIFRank_plus(text_fr):

    text = "Discrete output feedback sliding mode control of second order systems - a moving switching line approach The sliding mode control systems (SMCS) for which the switching variable is designed independent of the initial conditions are known to be sensitive to parameter variations and extraneous disturbances during the reaching phase. For second order systems this drawback is eliminated by using the moving switching line technique where the switching line is initially designed to pass the initial conditions and is subsequently moved towards a predetermined switching line. In this paper, we make use of the above idea of moving switching line together with the reaching law approach to design a discrete output feedback sliding mode control. The main contributions of this work are such that we do not require to use system states as it makes use of only the output samples for designing the controller. and by using the moving switching line a low sensitivity system is obtained through shortening the reaching phase. Simulation results show that the fast output sampling feedback guarantees sliding motion similar to that obtained using state feedback"


    #keyphrases = SIFRank(text_fr, SIF, en_model, N=25,elmo_layers_weight=elmo_layers_weight)
    keyphrases_ = SIFRank_plus(text_fr, SIF, en_model, N=15, elmo_layers_weight=elmo_layers_weight)

    #log_writing(f' Candidate keyphrase  SIFRank : {keyphrases}')
    log_writing(f' Candidate keyphrase  SIFRank_plus : {keyphrases_}')
    return keyphrases_


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

    path_output = "/Users/oaouina/Applications/brat_Annotator/data/processed_crh"
    # now we will extract from each document its important np
    count=0
    doc_section=[]
    for filename in os.listdir(path_output):
            list_core_sections=[]
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

                list_core_sections=get_core_section(fulltext)
                log_writing("############section beg ####################")
                for name_sct ,core_sct in list_core_sections:
                    print("beginning to extract NPs")
                    #extract sentences
                    sentences = extract_sentences(core_sct)
                    for i, sentence in enumerate(sentences, 1):
                        #now we will extract the keyphrases from sentences wich can cost in performances but it's
                        keyphrases = extract_SIFRank_plus(sentence)
                        # keyphrases_l contain keyphrases of a text
                        keyphrases_l.extend(keyphrases)

                keyphrases_only=keyphrases_l
                if len(keyphrases_l)>0:
                    unique_kp = []
                    [unique_kp.append(item) for item in keyphrases_l if item not in unique_kp]
                    keyphrases_only= [item for item,score in unique_kp if len(item) > 2 and score > 0.5 and '<rue />' not in item and '<nom />' not in item and "/>" not in item and "<" not in item ]
                list_matches=[]
                for kp in keyphrases_only:
                        list_matches.extend(find_chunks_with_regex(fulltext,kp))
                filtered_entities=keep_longest_overlapping_entities(list_matches)
                write_brat_annotation(file_path,filtered_entities)
             #print(list_matches)
            print(filename)

            #unique_strings = remove_included_strings(list_matches)
            print("Finaal results ############################")
                #print(unique_strings)





