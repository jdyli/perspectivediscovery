'''
Find most-probable topic per document, per topic model and save to csv-file for further analysis
The topic models are: random model, LDA, JST, TAM, LAM and VODUM
'''

import pandas as pd
import operator
from sklearn.metrics.cluster import adjusted_rand_score
import os

def evaluate_JST(jst_foldername, data, perspectives):
    number_of_topics = 3
    number_of_words = 10
    pi = '../data/models/JST/' + jst_foldername + '/final.pi'  # sentiment distribution
    theta = '../data/models/JST/' + jst_foldername + '/final.theta'  # topics distribution per sentiment
    twords = '../data/models/JST/' + jst_foldername + '/final.twords'  # sentiment, topic, words

    # Retrieving the sentiment labels per document
    pi_df = pd.DataFrame(columns=['sentiment_label', 'sentiment_contrib', 'sum_topic'])
    with open(pi) as pi_open:
        for line in enumerate(pi_open):
            sum_number = 1
            sentiment_label, sentiment_contrib = max(enumerate(line[1].split(' ')[2:]), key=operator.itemgetter(1))
            if int(sentiment_label) == 1:
                sum_number = 4
            pi_df = pi_df.append({'sentiment_label': sentiment_label, 'sentiment_contrib': sentiment_contrib,
                                  'sum_topic': sum_number}, ignore_index=True)
    pi_open.close()

    pi_df = pi_df.reset_index()
    pi_df['sentiment_label'] = pi_df['sentiment_label'].astype(int)
    pi_df.columns = ['doc_no', 'sentiment_label', 'sentiment_contrib', 'sum_topic']
    unique_sentiment = len(pi_df['sentiment_label'].unique()) + 1

    # Retrieving the topics per document and sentiment label
    theta_df = pd.DataFrame()
    number_of_articles = len(pi_df)  # for theta and twords
    for number in range(0, number_of_articles):
        position = (pi_df.iloc[number][0] * unique_sentiment) + 1 + pi_df.iloc[number][1]
        with open(theta) as theta_open:
            for index, line in enumerate(theta_open):
                if index == position:
                    line_split = line.split(' ')
                    topics_sorted = sorted(range(len(line_split)), key=lambda k: line_split[k])[::-1]
                    topic_keys = {}
                    for element in range(0, number_of_topics):
                        topic_keys[element] = topics_sorted[element]
                    theta_df = theta_df.append(topic_keys, ignore_index=True)
        theta_open.close()

    theta_df = theta_df.reset_index()
    theta_df.columns = ['doc_no', 'theta0', 'theta1', 'theta2']
    theta_df['theta0'] = theta_df['theta0'].astype(int)
    theta_df['theta1'] = theta_df['theta1'].astype(int)
    theta_df['theta2'] = theta_df['theta2'].astype(int)

    sentiment_topic_df = pd.merge(pi_df, theta_df, on='doc_no')
    sentiment_topic_df['sum_topic'] = sentiment_topic_df['sum_topic'] + sentiment_topic_df['theta0']
    # Retrieving the words per document, topic and sentiment label
    twords_df = pd.DataFrame()
    twords_open = open(twords, "r")
    read_twords = twords_open.read().splitlines()
    for number in range(0, number_of_articles):
        dictionary_sentiment_topic_words = {}
        for element in range(0, number_of_topics):
            name_of_line = 'Label' + str(sentiment_topic_df.iloc[number][1]) + '_Topic' + str(
                sentiment_topic_df.iloc[number][3 + element])
            subset_words = []
            for index, line in enumerate(read_twords):
                if line == name_of_line:
                    for k in range(1, number_of_words + 1):
                        subset_words.append(read_twords[index + k].split(' ')[0])
                    dictionary_sentiment_topic_words[element] = ' '.join([str(x) for x in subset_words])
        twords_df = twords_df.append(dictionary_sentiment_topic_words, ignore_index=True)
    twords_open.close()

    twords_df = twords_df.reset_index()
    twords_df.columns = ['doc_no', 'words0', 'words1', 'words2']
    final_results_df = pd.concat([data, twords_df['words0'], sentiment_topic_df[['sentiment_label', 'sum_topic', 'theta0']], perspectives], axis=1)
    final_results_df.columns = ['doc', 'words', 'sentiment_label', 'sum_topic', 'theta0', 'perspective']
    final_results_df.to_csv('../data/models/JST/' + jst_foldername + '/top_topic_per_doc.csv', index=False)
    return final_results_df

def evaluate_VODUM(vodum_foldername, data, perspectives, topics_list):
    vodum_df = pd.DataFrame(columns=['words', 'viewpoint', 'topic', 'sum_topic'])
    file_name_documents = '../data/models/VODUM/' + vodum_foldername + '/model-01-final.assign'

    with open(file_name_documents) as read_docs:
        for line in enumerate(read_docs):
            switch_number = 1
            viewpoint_label = int(line[1].split('|')[0])
            topic_label = int(line[1].split(';')[1].rstrip('\n'))
            if int(viewpoint_label) == 1:
                switch_number = 3 # Different from JST! here is switch number used
            sum_topic = int(viewpoint_label) + switch_number + int(topic_label)
            vodum_df = vodum_df.append({'words': topics_list[sum_topic-1], 'viewpoint': viewpoint_label, 'topic': topic_label, 'sum_topic': sum_topic}, ignore_index=True)
    read_docs.close()
    final_results_df = pd.concat([data, vodum_df, perspectives], axis=1)
    final_results_df.columns = ['doc', 'words', 'viewpoint', 'topic', 'sum_topic', 'perspective']
    final_results_df.to_csv('../data/models/VODUM/' + vodum_foldername + '/top_topic_per_doc.csv', index=False)
    return final_results_df

def evaluate_LDA(ldafoldername, corpus, ldamodel, top_topics_n, data, perspectives):
    filename = '../data/models/LDA/' + ldafoldername + '/top_topic_per_doc.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    lda_df = pd.DataFrame(columns=['words', 'topic'])

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list # row = document = [(topic number, probability), (t,p), ...]
        row = sorted(row, key=lambda x: (x[1]), reverse=True) # sort so the topics with highest probability come first
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for element in range(0, len(row[:top_topics_n])):
            wp = ldamodel.show_topic(row[element][0])
            topic_keywords = ", ".join([word for word, prop in wp])
            lda_df = lda_df.append({'words': topic_keywords, 'topic': row[0][0] + 1}, ignore_index=True)
    # results_lda_df = sent_topics_df.reset_index()
    final_results_df = pd.concat([data, lda_df, perspectives], axis=1)
    final_results_df.columns = ['doc', 'words', 'sum_topic', 'perspective']
    final_results_df.to_csv(filename, index=False)
    return final_results_df

def evaluate_TAM(tam_foldername, data, perspectives, topics_list):
    file_name_documents = '../data/models/TAM/' + tam_foldername + '/tokenized_data.txt.assign'
    topic_df = pd.DataFrame(columns=['words', 'topic'])

    # Get main topic in each document
    with open(file_name_documents) as topic_open:
        for line in enumerate(topic_open):
            splitted_line = line[1].split(' ')
            total_topic_numbers = []
            for token in splitted_line[1:len(splitted_line)-1]:
                total_topic_numbers.append(int(token.split(':')[2]))
            # count and choose the biggest topic number
            topic_majority = max(total_topic_numbers, key=total_topic_numbers.count)
            topic_df = topic_df.append({'words': topics_list[topic_majority], 'topic': topic_majority}, ignore_index=True)

    final_results_df = pd.concat([data, topic_df, perspectives], axis=1)
    final_results_df['sum_topic'] = final_results_df['topic'] + 1
    final_results_df.columns = ['docs', 'words', 'topic', 'perspective', 'sum_topic']
    final_results_df.to_csv('../data/models/TAM/' + tam_foldername + '/top_topic_per_doc.csv', index=False)
    return final_results_df

def evaluate_LAM(lam_foldername, data, perspectives, topics_dict):
    topic_doc = '../data/models/LAM/' + lam_foldername + '/lam.doc_topic_distribution'
    vp_doc = '../data/models/LAM/' + lam_foldername + '/lam.topic_viewpoint_distribution'
    topic_df = pd.DataFrame(columns=['topic'])
    vp_df = pd.DataFrame(columns=['viewpoint'])

    with open(topic_doc) as topic_open:
        for line in enumerate(topic_open):
            if 'DocId' not in line[1]:
                topic_list = line[1].split('\t')[1:]
                sorted_topics = sorted(range(len(topic_list)), key=lambda k: topic_list[k])[::-1]
                topic_df = topic_df.append({'topic': sorted_topics[0]}, ignore_index=True)
    topic_open.close()

    # Retrieving the viewpoint per document and topic label
    number_of_articles = len(topic_df)
    for number in range(0, number_of_articles):
        position = int(topic_df.iloc[number][0])
        with open(vp_doc) as vp_open:
            for index, line in enumerate(vp_open):
                line_split = line.split('\t')
                if line_split[0] == str(position):
                    vp_label, vp_contrib = max(enumerate(line_split[1:]), key=operator.itemgetter(1))
                    vp_df = vp_df.append({'viewpoint': vp_label}, ignore_index=True)
        vp_open.close()

    viewpoint_results_df = pd.concat([topic_df, vp_df], axis=1)
    viewpoint_results_df['words'] = ""
    for row in range(0, len(viewpoint_results_df)):
        topic = viewpoint_results_df.iloc[row]['topic']
        viewpoint = viewpoint_results_df.iloc[row]['viewpoint']
        viewpoint_results_df.at[row, 'words'] = topics_dict['t' + str(topic) + 'vp' + str(viewpoint)]

    final_results_df = pd.concat([data, viewpoint_results_df, perspectives], axis=1)
    final_results_df['sum_topic'] = final_results_df['topic'] + 1
    final_results_df.columns = ['docs', 'topic', 'viewpoint', 'words', 'perspective', 'sum_topic']
    final_results_df.to_csv('../data/models/LAM/' + lam_foldername + '/top_topic_per_doc.csv', index=False)
    return final_results_df

# ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)
def rand_index(ground_truth, prediction):
    score = adjusted_rand_score(ground_truth, prediction)
    return score


