import gensim

def fit_lda_model(topics_n, corpus, dictionary):
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=dictionary,
                                            num_topics=topics_n,
                                            iterations=1000,
                                            alpha=50/topics_n,
                                            eta=0.01,
                                            per_word_topics=True)

    # Print the Keyword in the 10 topics
    topics = lda_model.print_topics()
    print('------- LDA ------- \n', topics)
    return lda_model

# def get_top_topic_words(corpus, ldamodel, top_topics_n):
#     # Init output
#     sent_topics_df = pd.DataFrame()
#     # Get main topic in each document
#     for i, row_list in enumerate(ldamodel[corpus]):
#         row = row_list[0] if ldamodel.per_word_topics else row_list # row = document = [(topic number, probability), (t,p), ...]
#         row = sorted(row, key=lambda x: (x[1]), reverse=True) # sort so the topics with highest probability come first
#         # Get the Dominant topic, Perc Contribution and Keywords for each document
#         list_of_keywords = {}
#         for element in range(0, len(row[:top_topics_n])):
#             wp = ldamodel.show_topic(row[element][0])
#             topic_keywords = ", ".join([word for word, prop in wp])
#             list_of_keywords[element] = topic_keywords
#         sent_topics_df = sent_topics_df.append(list_of_keywords, ignore_index=True)
#
#     # Format
#     results_lda_df = sent_topics_df.reset_index()
#     results_lda_df.columns = ['doc_no', 'LDA_t0', 'LDA_t1', 'LDA_t2']
#     return(results_lda_df)

