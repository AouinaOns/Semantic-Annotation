# Semantic-Annotation
# Inter-Annotator Agreement Evaluation

We evaluated the inter-annotator agreement through the contributions of two annotators across all tasks. This agreement was found to be 0.79, indicating significant consistency in the annotations provided. Consequently, we proceeded to consolidate all annotated reports. Named Entity Recognition (NER) and context extraction of concepts demonstrated an overall precision of 0.9610, recall of 0.9248, and an F1 score of 0.9425. Detailed results are presented in Table 1 below.

## Table 1: Quantitative Results of Named Entity Recognition Evaluation

| **Entité Nommée**          | **Total** | **Précision** | **Rappel** | **F1-score** |
|----------------------------|-----------|---------------|------------|--------------|
| **Catégories Générales**   |           |               |            |              |
| ÉpisodeDeSoin              | 400       | 0.8321        | 0.9485     | 0.8865       |
| ÉvénementVécu              | 343       | 0.9589        | 0.9333     | 0.9459       |
| ExamenClinique             | 308       | 0.9666        | 0.8811     | 0.9219       |
| Hospitalisation            | 520       | 0.9245        | 0.8124     | 0.8648       |
| Individu                   | 540       | 0.9910        | 0.7774     | 0.8713       |
| Maladie                    | 1166      | 0.9024        | 0.8301     | 0.8502       |
| PartieDuCorps              | 22        | 0.9800        | 0.6667     | 0.7912       |
| Qualité                    | 600       | 0.9882        | 0.8802     | 0.9317       |
| SigneClinique              | 1250      | 0.9014        | 0.8524     | 0.8762       |
|                            |           |               |            |              |
| **Médicaments**            |           |               |            |              |
| Substance                  | 1034      | 0.9805        | 0.8200     | 0.8933       |
| Dosage                     | 608       | 0.8100        | 0.6800     | 0.7406       |
| Forme pharmaceutique       | 14        | 0.9130        | 0.8570     | 0.8844       |
|                            |           |               |            |              |
| **Informations temporelles** |         |               |            |              |
| Date                       | 1208      | 0.9142        | 0.9209     | 0.9176       |
| Durée                      | 221       | 0.7481        | 0.9019     | 0.8177       |
| Fréquence                  | 511       | 0.9254        | 0.7688     | 0.8405       |
| Time (Moment)              | 88        | 0.8750        | 0.9459     | 0.9091       |


## Table 2: Quantitative Results of Relation Extraction

| Relation                  | Total | Precision | Rappel | F1     |
|---------------------------|-------|-----------|--------|--------|
| Relations Temporelles     | 860   | 0.9130    | 0.8077 | 0.8571 |
| A pour motif              | 1050  | 0.9750    | 0.9070 | 0.9398 |
| Participe                 | 286   | 0.9831    | 0.5800 | 0.7296 |
| Qualifie                  | 756   | 0.9211    | 0.7368 | 0.8188 |
| Relations Medicaments dosage | 521 | 0.9046    | 0.8333 | 0.8678 |

