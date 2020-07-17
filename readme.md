# Perspective Discovery in Controversial Debates

## Introduction
This repository is part of my master thesis at TU Delft. 

### Contributions
With this repository we evaluate five existing topic models and evaluate their effectiveness
using topic coherence, adjusted rand index and user study results.
We provide an annotated dataset with perspective labels on forum post entries.
The main research question is: What topic model is able to discover human understandable perspectives on controversial issues?

## Functionalities
This repository provides the following:
- An annotated dataset of opinionated documents with each document a set of perspective labels based on www.abortion.procon.org
(This dataset is one where perspective 16+17 on ProCon has been combined due to higher annotation reliability)
- Preprocessing functionalities to transform raw documents into a desired set of tokens for the topic model input. 
(Some of the possible preprocessing functions are: removing non-sentiment words, n-grams, outlier removal or antonym transformations)
- Transform the tokenized documents into the correct data format for the used topic models
- An evaluative framework to compute the adjusted rand index and topic coherence score 
of the topic models
- User study results on the level of human understandability of topic model's output
- For now the evaluations only work with models: LDA, VODUM, TAM, LAM and JST

All code has been run on ```Python 3.7```

## User guide
Below you see the folder structure (explicit files are excluded).
```
.
+-- data     
|   +-- abortion_debateorg
|   +-- evaluate_annotations
|   +-- results1
|   +-- topicmodel_templates
+-- userstudy
+-- src
```

- The raw data is saved under ```./data/abortion_debateorg/```. File ```./src/main.py``` lets you
preprocess the tokens and evaluate the topic models according to two metrics: topic coherence and adjusted rand index.
- We have annotated the raw data which has been evaluated by two independent annotators. We
then computed the reliability with the Krippendorff's Alpha. These reliability score results can be found under ```./data/evaluate_annotations/```.
- Note that running the topic models themselves is not part of this code. The code does provide the correct tokenized format to make it run on 
the topic models described above. This means that you have to use the external source code of the existing topic models to run tokenized formats and let it direct to ```./data/results1/[TOPIC_MODEL_NAME/UNIQUE_FOLDERNAME]/```. Using
```./src/main.py``` you can run these results to compute the evaluation metrics. after you have tokenized the documents it will be saved under ```./data/results1/[TOPIC_MODEL_NAME/UNIQUE_FOLDERNAME]/```
as csv-file or txt-file (depending on the type of model). Under ```./data/results1/[TOPICMODEL]/example``` we provide for each topic model an example of what data is exactly stored per model. 
You find the tokenized input data that uses baseline tokenization and the dataset ```abortion_debateorg_COMPLETE.csv``` as well as the corresponding topic model's output data. 
You can also use this example to run the main file ```./src/main.py```.
- Under ```/data/topicmodel_templates/``` we provide a template as to how you can configure the required parameter file of the topic models VODUM, TAM, LAM and JST. 
- ```./userstudy``` provide results of the user study


If you have any questions, feel free to contact me at any time!

## Sources of the external topic models
- VODUM: https://github.com/tthonet/VODUM 
- TAM: https://github.com/blade091shenwei/TAM_ccLDA
- JST: https://github.com/linron84/JST (go to the Debug folder to run their code)
- LAM: https://github.com/aghie/lam 
