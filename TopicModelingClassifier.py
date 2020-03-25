from nltk.tag.stanford import StanfordPOSTagger
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import FrenchStemmer
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
from unidecode import unidecode
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from corextopic import corextopic as ct

import warnings
warnings.filterwarnings("ignore")

class TopicModelingClassifier:
    
    def __init__(self, corpus, n_topics):

        self.corex_model = self.CorExModel(
            corpus = corpus, 
            n_topics = n_topics
        )

        '''
        if supervised :
            self.classifier = self.SupervisedClassifier()

        else :
            self.classifier = self.UnsupervisedClassifier()
        '''

    def fit(self):
        
        pass 

    def transform(self):

        pass 

    def predict(self):
        # Depending on the chosen method for the classifier, it will return a list of probabilites or an unique scalar value
        pass 

    def visualize(self):
    
        pass 

    class CorExModel: 
   
        def __init__(self, corpus, n_topics, stem = False, anchors = None, anchor_strength = None, process = False, verbose = False):
            '''
            Description:
            
                Class constructor
                
            Parameters:
            
                - corpus (List[String]) : the list of raw descriptions.
                - n_topics (Integer) : the default number of topics you're looking for. The parameter could be changing with the @tune() method.
                - anchors (List[String] | List[List[String]]) : chosen anchors for the CorEx models. Anchors should be specific to destinations or at least to clusters of destinations.
                - anchor_strength (Integer) : the weigth given to anchors.
                - process (String) : specify if the corpus need to be processed using @process method.
                - verbose (Boolean) : specify if we want information about the model while it is training.
                - stem (Boolean) : specify if we want to use stemming in the processing step
            '''
            
            self.corpus = corpus
            self.n_topics = n_topics
            self.model = ct.Corex(n_hidden = self.n_topics, anchors = anchors, anchor_strength = anchor_strength, verbose = verbose, process = process, seed=42)
            self.is_fitted = False

            if process :
                self.train_data = self.process_corpus(stem = stem)
            else : 
                self.train_data = self.corpus 
            
            self.vectorizer = TfidfVectorizer(
                    max_df=.7,
                    min_df=.01,
                    max_features=None,
                    ngram_range=(1, 2),
                    norm=None,
                    binary=True,
                    use_idf=True,
                    sublinear_tf=False
            )

        @classmethod
        def tune(cls, corpus, stem = False, anchors = None, anchor_strength = None, process = False, verbose = False):
            '''
            Description :
                
                Optional method aiming to find the best n_topics for the CorEx model. This method is used when one wants to use this model as a first layer (projection onto the space of topic membership probabilities) of a classifier model.
                
            Parameters : 
                
                - None
            '''

            # Create a fake model with an arbitrary high number of hidden topics
            test_model = cls(corpus = corpus, n_topics = 100, stem = stem, anchors = anchors, anchor_strength = anchor_strength, process = process, verbose = False)
            test_model.fit()

            # Find the optimal number of topics. This will be done filtering all topics which total correlation is lower than 5% of the max total correlation. 
            optimal_n_topics = len(list(filter(lambda x : x >= 0.05 * test_model.model.tcs.max(), test_model.model.tcs)))

            # Return an optimal model to be used as a input for a further topic modeling extraction
            return cls(corpus = corpus, n_topics = optimal_n_topics, stem = stem, anchors = anchors, anchor_strength = anchor_strength, process = process, verbose = False)

        def fit(self):
            '''
            Description :
                
                Train the CorexModel on the input data. The Topic Modeling algorithm will be trained on a specific destination

            Parameters : 

                - None
            '''
        
            if not self.check_is_fitted() :
            
                self.vectorizer = self.vectorizer.fit(self.train_data)
                tfidf = self.vectorizer.transform(self.train_data)
                vocab = self.vectorizer.get_feature_names()

                self.model = self.model.fit(tfidf, words = vocab)
                self.is_fitted = True 
            
            else :
                print('Model already fitted')
        
        def transform(self, display_mode, dataframe_mode = True):
            '''
            Description :
                
                Apply topic attribution for every description in the training set.

            Parameters :

                - display_mode (String) : ['proba' or 'labels'] specify if the output should display probabilities that each document belongs to each topic, or just binary classification using Softmax.
            ''' 

            if display_mode not in ['proba', 'labels'] :
                raise AttributeError("Select a desired display_mode between 'proba' and 'labels'")

            if not self.check_is_fitted():
                self.fit() 
                self.is_fitted = True 

            if display_mode == 'proba' :
                if dataframe_mode :
                    return pd.DataFrame(self.model.p_y_given_x, columns = ['Topic #%i'%i for i in range(1, self.n_topics + 1)], index = ['Desc %i'%i for i in range(1, len(self.train_data) + 1)])
                
                else : 
                    return self.model.p_y_given_x

            else :
                if dataframe_mode :
                    return pd.DataFrame(self.model.labels, columns = ['Topic #%i'%i for i in range(1, self.n_topics + 1)], index = ['Desc %i'%i for i in range(1, len(self.train_data) + 1)])
                
                else :
                    return self.model.labels

        def predict_proba(self, input_description, stem = False):
            '''
            Description :

                Predict, for each topic, the probability that the input description belongs to this topic.

            Parameters : 

                - input_description (String) : The input description we want to predict topic memberships.
                - stem (String) : Specify if stemming must be applied while processing the input_description.
            '''

            if not self.check_is_fitted() :
                raise EnvironmentError("Your model has not been trained yet ! Do it first")
            
            processed_input_description = self.process_input(input_description = input_description, stem = stem)
            vectorized_input_description = self.vectorizer.transform([processed_input_description])
            proba_preds = list(self.model.predict_proba(vectorized_input_description)[0][0])

            return proba_preds
        
        def predict_labels(self, input_description, stem = False):
            '''
            Description :

                Predict, for each topic, the attributed topics given their probabilities of membership.

            Parameters : 

                - input_description (String) : The input description we want to predict topic memberships.
                - stem (String) : Specify if stemming must be applied while processing the input_description.
            '''

            if not self.check_is_fitted() :
                raise EnvironmentError("Your model has not been trained yet ! Do it first")
            
            proba_preds = self.predict_proba(input_description = input_description, stem = stem)
            softmax_values = softmax(proba_preds)

            labels = list(map(lambda x : x > .1, softmax_values))
            
            return labels

        def score(self):
            '''
            Description :

                Returns the total correlation of the model 

            Parameters :

                - None
            '''

            return self.model.tc 

        def visualize(self, n_words = 10):
            '''
            Description :

                Render a summary of the output of the model in the form of a pandas DataFrame.

            Parameters : 

                - n_words (Integer) : specify the number of generative words per topic to display.
            '''

            if self.check_is_fitted() :
                res = {}
                for i, topic_ngrams in enumerate(self.model.get_topics(n_words=n_words)):
                    topic_ngrams = [ngram[0] for ngram in topic_ngrams if ngram[1] > 0]
                    if not len(topic_ngrams) == n_words :
                        topic_ngrams += ['']*(n_words - len(topic_ngrams))

                    res["Topic #{}".format(i+1)] = topic_ngrams

                return pd.DataFrame(res, index = ['Word %s'%i for i in range(1, 11)])

            else:
                raise EnvironmentError("Your model has not been trained yet ! Do it first")

        def process_input(self, input_description, stem = False, undesired_words = []): 
            '''
            Description :
                
                Apply the processing steps to an input text

            Parameters : 
                
                - input_description (String) : the text to process.
                - stem (Boolean) : specify if the model should use stemmed descriptions or not.
                - undesired_words (List[String]) : list of all words that we want to filter in the processing step
            '''

            tokenizer_ = RegexpTokenizer(r'[^\d\W]+')
            stemmer_ = FrenchStemmer()
            lemmatizer_ = FrenchLefffLemmatizer()
            stop_words_ = stopwords.words('french')

            tokens = list(map(lambda x : x.lower(), tokenizer_.tokenize(input_description)))
            tokens = list(filter(lambda x : x not in stop_words_ and len(x) >= 3 and x not in undesired_words, tokens))
            tokens = [unidecode(lemmatizer_.lemmatize(token)) for token in tokens]

            if stem :
                return ' '.join(list(map(lambda x : stemmer_.stem(x), tokens)))
                
            else :
                return ' '.join(tokens)

        def process_corpus(self, stem = False):
            '''
                Description : 

                    Update the train dataset parameter (@train_data) by processing the input @corpus.

                Parameters :

                    - stem (Boolean) : specify if the model should use stemmed descriptions or not.
            '''

            res = []
            for description in self.corpus :
                processed_description = self.process_input(input_description = description, stem = stem)
                if not isnan(processed_description): 
                    res.append(processed_description)

            return res
                
        def check_is_fitted(self) :
            '''
            Description :

                Check if the model has already been fitted

            Parameters :

                - None
            '''

            return self.is_fitted
                    
    class SupervisedClassifier:

        def __init__(self, method, n_clusters):
            '''
            Description:
            
                Class constructor
                
            Parameters:
            
                - method (String) : You can choose between SOFT or HARD clustering. SOFT option will involve a simple K-Means algorithm, while HARD will use a GMM model. Naive Bayes Classifier could be used too.
                - n_clusters (Integer) : The desired number of clusters. By default it will be set to CorExModeling.n_topics.

            '''

            '''
            if method == 'SOFT' :
                self.model = KMeans('...')
            elif method == 'HARD' :
                self.model = GMM('...')

            else: 
                self.model = NaiveBayesClassifier('...')
            '''

            pass 
            
        
        def fit(self):

            pass 
        
        def transform(self):

            pass 

        def predict(self):

            pass 

    class UnsupervisedClassifier: 

        def __init__(self, train_data,  n_topics, model_choice = 'CorEx') :
            '''
            Description : 

                Class constructor 

            Parameters : 

                - train_data (List[List[Float]]) : the output probabilities for (a) document(s) to belong to each topic. 
                                                   This will be the output of CorExModel.transform() for the whole corpus or CorExModel.predict_proba() for a unique entry.

                - model_choice (String) : the model we want to use for the unsupervised classifier. You can either choose between the use of :

                                                + a Hierarchical Corex Model which will use the output of the first model of CorExModel.tune() (high dimension) 
                                                  as an input and will output a multi-label classification array in lower dimension.

                                                + a simple KMeans (for now) which will use the output of a first CorExModel.tune() (high dimension) and create clusters based on
                                                  the vectorized version of each description in the topic membership probabilities space.
            '''

            self.train_data = train_data 

            if model_choice not in ['CorEx', 'KMeans'] :
                raise TypeError("Wrong model choice, please choose between 'CorEx' and 'KMeans'.")
            
            elif model_choice == 'CorEx' :
                self.model = ct.Corex(n_hidden = n_topics)            

            elif model_choice == 'KMeans' : 
                self.model = KMeans(n_clusters = n_topics, random_state = 42)

        def fit(self):
            '''
            Description :

                Method used to fit the model to the input_data.
                
            Parameters : 

                - None
            '''

            self.model.fit(self.train_data)

        def transform(self):

            return self.model.transform()