'''
Main code of the project. Use the parameters below (line 210) to change the dataset. This file has two main tasks:
1) Lets you preprocess the raw documents into the desired tokenized documents. These are then saved in the correct
topic model format as csv-file or txt-file so you can run them on existing topic models from an external source.
- Modify clean_and_save_data() to choose your desired preprocessing pipeline on the raw documents.
- Files are saved under the './data/results1/[TOPIC_MODEL_NAME]/[UNIQUE_FOLDERNAME]/'

2) Lets you evaluate topic models based on aRI and topic coherence.
- This only works AFTER you've obtained the output of the topic models from the external source
- This works if you've put the output in the correct folder >> './data/results1/[TOPIC_MODEL_NAME]/[UNIQUE_FOLDERNAME]/'

Code works on the following topic models: LAM, TAM, JST, VODUM and LDA.
'''

import pandas as pd
import src.preprocess_data as preprocess
import src.build_lda as lda
from ast import literal_eval
from operator import itemgetter
from gensim.models import CoherenceModel
import src.data_formats as fm
import os.path
import nltk
import src.evaluate_models as evalmodels
import src.build_randommodel as randommodel

pd.set_option('display.max_colwidth', -1)


class Topicmodel(object):
    def __init__(self, foldername, topics_n, data, perspectives, bool_tokenize_docs):
        self.foldername = foldername
        self.topics_n = topics_n
        self.data = data
        self.min_topics_n = topics_n
        self.perspectives = perspectives
        self.bool_tokenize_docs = bool_tokenize_docs
        self.tokenized_data = None
        self.corpus = None
        self.dictionary = None
        self.preprocessed_data = None

    '''Preprocess the raw documents, tokenize and formats the tokenized documents according to topic model type
    Modify this clean_and_save_data() to your own liking'''

    def clean_and_save_data(self):
        self.preprocessed_data = preprocess.text_correction(self.data)
        cleaned_tokenized_data = preprocess.tokenize(self.preprocessed_data)
        # antonyms_data = preprocess.antonyms(cleaned_tokenized_data)   # UNCOMMENT IF YOU LIKE TO USE THIS PREPROCESSING FUNCTION
        # sentiment_data = preprocess.remove_nonsentiments(cleaned_tokenized_data)
        # neg_data = preprocess.negation(cleaned_tokenized_data)
        # bigram_data = preprocess.bigrams_and_unigrams(cleaned_tokenized_data)
        # tribiunigram_data = preprocess.trigrams_bigrams_unigrams(cleaned_tokenized_data)
        # extreme_removal_data = preprocess.remove_extremes(cleaned_tokenized_data)
        # zipslaw_data = preprocess.zipslaw(data=cleaned_tokenized_data, frequentp=0.005)

        self.tokenized_data = cleaned_tokenized_data
        self.corpus, self.dictionary = preprocess.prepare_model(self.tokenized_data)
        if self.bool_tokenize_docs == True:
            fm.create_dataformat1(self.tokenized_data, self.foldername, 'JST')
            fm.create_dataformat1(self.tokenized_data, self.foldername, 'TAM')
            fm.create_dataformat2(self.tokenized_data, self.foldername, 'VODUM')
            fm.create_dataformat4(self.perspectives, self.data, self.tokenized_data, self.foldername, 'LAM')

    '''Evaluate all the topic models after you have retrieved the results from an external source'''

    def evaluate_allmodels(self, file_extension, bool_allmodels):
        print('Number of topics: %d' % self.topics_n)
        print('Number of documents: %d' % len(self.corpus))
        print('Number of unique tokens: %d' % len(self.dictionary))
        topicmodel_list = ['JST', 'VODUM', 'TAM', 'LAM']
        coherence_list = []
        randindex_list = []
        output_list = []

        for topic_number in range(self.min_topics_n, self.topics_n + 1):
            print(topic_number)
            lda_model = lda.fit_lda_model(topic_number, self.corpus, self.dictionary)
            lda_topics = lda_model.show_topics(formatted=False)
            lda_topics = [[word for word, prob in topic] for topicid, topic in lda_topics]
            lda_cm = CoherenceModel(topics=lda_topics, texts=self.tokenized_data, corpus=self.corpus,
                                    dictionary=self.dictionary, coherence='c_v')
            lda_coherence = lda_cm.get_coherence()
            results_lda_df = evalmodels.evaluate_LDA(ldafoldername=file_extension, corpus=self.corpus,
                                                     ldamodel=lda_model, top_topics_n=1,
                                                     data=self.data, perspectives=self.perspectives)
            lda_randindex = evalmodels.rand_index(results_lda_df['perspective'], results_lda_df['sum_topic'])
            print('LDA : \n', 'Adjust. Rand index: ', lda_randindex, '\n Topic coherence: ', lda_coherence)
            print(lda_topics)

        if bool_allmodels == True:
            '''JST'''
            jstfile = '../data/results1/JST/' + file_extension + '/final.twords'
            if os.path.isfile(jstfile):
                jst_topic = fm.read_jst_toptopics(jstfile)
                jst_results_model_df = evalmodels.evaluate_JST(file_extension, self.data, self.perspectives)
                coherence_list.append(CoherenceModel(topics=jst_topic, texts=self.tokenized_data, corpus=self.corpus,
                                                     dictionary=self.dictionary, coherence='c_v').get_coherence())
                randindex_list.append(
                    evalmodels.rand_index(jst_results_model_df['perspective'], jst_results_model_df['sum_topic']))
                output_list.append(jst_topic)

            '''VODUM'''
            vodum_topic = fm.read_vodum_toptopics(file_extension)
            vodum_results_model_df = evalmodels.evaluate_VODUM(vodum_foldername=file_extension, data=self.data,
                                                               perspectives=self.perspectives, topics_list=vodum_topic)
            coherence_list.append(CoherenceModel(topics=vodum_topic, texts=self.tokenized_data, corpus=self.corpus,
                                                 dictionary=self.dictionary, coherence='c_v').get_coherence())
            randindex_list.append(
                evalmodels.rand_index(vodum_results_model_df['perspective'], vodum_results_model_df['sum_topic']))
            output_list.append(vodum_topic)

            '''TAM'''
            tamfile = '../data/results1/TAM/' + file_extension + '/output_topwords_tokenized_data.txt'
            tam_topic = fm.read_tam_toptopics(tamfile)
            tam_results_model_df = evalmodels.evaluate_TAM(tam_foldername=file_extension, data=self.data,
                                                           perspectives=self.perspectives, topics_list=tam_topic)
            coherence_list.append(CoherenceModel(topics=tam_topic, texts=self.tokenized_data, corpus=self.corpus,
                                                 dictionary=self.dictionary, coherence='c_v').get_coherence())
            randindex_list.append(
                evalmodels.rand_index(tam_results_model_df['perspective'], tam_results_model_df['sum_topic']))
            output_list.append(tam_topic)

            '''LAM'''
            lamfile = '../data/results1/LAM/' + file_extension + '/lam.out'
            lam_topic_dict, lam_topic = fm.read_lam_toptopics(lamfile)
            lam_results_model_df = evalmodels.evaluate_LAM(lam_foldername=file_extension, data=self.data,
                                                           perspectives=self.perspectives, topics_dict=lam_topic_dict)
            coherence_list.append(CoherenceModel(topics=lam_topic, texts=self.tokenized_data, corpus=self.corpus,
                                                 dictionary=self.dictionary, coherence='c_v').get_coherence())
            randindex_list.append(
                evalmodels.rand_index(lam_results_model_df['perspective'], lam_results_model_df['sum_topic']))
            output_list.append(lam_topic)

            '''RANDOM TF-IDF'''
            random_model, top_tokens = randommodel.fit_tfidf(self.tokenized_data)
            print('RANDOM MODEL results: \n', random_model)

            for i in range(0, len(topicmodel_list)):
                print('\n ----------', topicmodel_list[i], ' ------------- \n Adjust. Rand index: '
                      , randindex_list[i], '\n Topic coherence: ', coherence_list[i])
                print('Model output:')
                for topicelement in output_list[i]:
                    print(topicelement)


'''Retrieve raw data and create balanced sample of 600 documents'''


def get_data(domain):
    perspectives_n = 6
    articles_n = 100

    if 'fullcorpus_abortion' in domain:
        abortion_file = '../data/abortion_debateorg/fullcorpus_abortion.csv' # change path if necessary
        data_df = pd.read_csv(abortion_file)
        data_df = data_df.dropna()
        data_df['perspective_list'] = data_df['perspective'].apply(literal_eval)
        data_df['perspective_n'] = data_df['perspective_list'].str.len()
        data_df = data_df.sort_values(by=['perspective_n'])
        data_df = data_df.loc[(data_df['perspective_n'] == 1) & (data_df['perspective'] != '[31]')
                              & (data_df['perspective'] != '[32]')]
        topics_all = []
        newsubset = pd.DataFrame()

        perspectives = list([a for b in data_df.perspective_list.tolist() for a in b])
        fdist = nltk.FreqDist(perspectives)
        common_perspectives = fdist.most_common(perspectives_n)
        common_perspectives = sorted(common_perspectives, key=itemgetter(1))
        for item in common_perspectives:
            topics_all.append(item[0])
            mask = data_df.perspective_list.apply(lambda x: item[0] in x)
            newsubset = newsubset.append(data_df[mask].sample(n=articles_n, random_state=1))
        newsubset.drop_duplicates(subset='web-scraper-order', keep='first', inplace=True)
        newsubset = newsubset.reset_index(drop=True)

    else:
        topics_all = [2, 3, 11, 16, 18, 24]
        abortion_file = '../data/abortion_debateorg/finalcorpus600_abortion.csv' # change path if necessary
        newsubset = pd.read_csv(abortion_file)
        newsubset['perspective_list'] = newsubset['perspective'].apply(literal_eval)
    data = newsubset['article']
    topics_n_all = len(topics_all)
    perspectives_data = newsubset['perspective_list']
    '''end evenly distributed'''
    # newsubset.to_csv('../data/abortion_debateorg/newsubset_abortion.csv', index=False) # UNCOMMENT TO SAVE
    return data, topics_n_all, perspectives_data


def menu(domain, foldername, bool_allmodels, bool_tokenize_docs):
    data, topics_n_all, perspectives = get_data(domain)
    print('EVALUATING MODELS WITH ALL-DATA .....')
    if foldername == '' and bool_allmodels == True:
        print('No foldername is given. Try again')
        exit()
    tm = Topicmodel(foldername, topics_n_all, data, perspectives, bool_tokenize_docs)
    tm.clean_and_save_data()
    if not bool_tokenize_docs:
        tm.evaluate_allmodels(file_extension=foldername, bool_allmodels=bool_allmodels)


'''PARAMETERS'''
# domain = 'fullcorpus_abortion'
domain = 'finalcorpus600_abortion' # this file has been used for the research

print('Do you want to prepare your raw documents and tokenize them for the topic models? (y/n)') # choose this option if you haven't tokenized documents yet
tokenize_docs = input()
if tokenize_docs == 'y':
    print('How should the folder be called to save this data? The folder will be saved in [./data/results1]')
    foldername = input()
    allmodels = False
    tokenize_docs = True
else:
    print(
        'Do you want to evaluate all the five topic models? (y/n) [YOU CAN ONLY DO THIS WHEN YOU HAVE ALREADY COMPUTED RESULTS FROM THE EXTERNAL SOURCE]')
    allmodels = input()
    if allmodels == 'y':
        print(
            'In which folder are these results saved? [./data/results1]? [CHOOSE THE NAME WHERE YOU HAVE SAVED THE TOKENIZED DOCS AND THE MODEL RESULTS] ')
        foldername = input()
        allmodels = True
        tokenize_docs = False
    else:
        print('In that case only LDA will be computed ....')
        allmodels = False
        tokenize_docs = False
        foldername = ''

menu(domain, foldername, allmodels, tokenize_docs)
