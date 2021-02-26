import re
import pandas as pd
import bs4
import pickle
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, PunktSentenceTokenizer
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

import networkx as nx
from spacy.matcher import Matcher 
from spacy.tokens import Span 

import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.feature_extraction.text import CountVectorizer 

def hotel_list():
    hotel_list = pd.read_csv("../data/Hotel_Reviews.csv", usecols=["Hotel_Name", "Positive_Review"])

    selected = hotel_list.loc[hotel_list['Hotel_Name'] == 'Hotel Arena']
    selected.to_csv("../selected.csv", index=False, header=None)
    print(len(hotel_list.Hotel_Name.unique()))

def get_entities(sent):
	## chunk 1
	ent1 = ""
	ent2 = ""

	prv_tok_dep = ""    # dependency tag of previous token in the sentence
	prv_tok_text = ""   # previous token in the sentence

	prefix = ""
	modifier = ""

	sent = sent.strip()
  #############################################################
  
	for tok in nlp(sent):
		print(tok.text, "...", tok.dep_)

		# tok.text :: the actual word token
		# tok.dep_ :: the label (dependency tag)

		## chunk 2
		# if token is a punctuation mark then move on to the next token
		if tok.dep_ != "punct":
			# check: token is a compound word or not
			if tok.dep_ == "compound":
				prefix = tok.text
			# if the previous word was also a 'compound' then add the current word to it
			if prv_tok_dep == "compound":
				prefix = prv_tok_text + " "+ tok.text
			
			# check: token is a modifier or not
			if tok.dep_.endswith("mod") == True:
				modifier = tok.text
			# if the previous word was also a 'compound' then add the current word to it
			if prv_tok_dep == "compound":
				modifier = prv_tok_text + " "+ tok.text
			
			## chunk 3
			if tok.dep_.find("subj") == True:
				ent1 = modifier +" "+ prefix + " "+ tok.text
				prefix = ""
				modifier = ""
				prv_tok_dep = ""
				prv_tok_text = ""      

			## chunk 4
			if tok.dep_.find("obj") == True:
				ent2 = modifier +" "+ prefix +" "+ tok.text
			
			## chunk 5  
			# update variables
			prv_tok_dep = tok.dep_
			prv_tok_text = tok.text
			print
			# print("prv_tok_dep :: {}".format(prv_tok_dep))
			# print("prv_tok_text :: {}\n".format(prv_tok_text))
  #############################################################
	return [ent1.strip(), ent2.strip()]

def get_entities2(sent):
	## chunk 1
	ent1 = ""
	ent2 = ""

	prv_tok_dep = ""    # dependency tag of previous token in the sentence
	prv_tok_text = ""   # previous token in the sentence

	prefix = ""
	modifier = ""
	preposition = ""

	sent = sent.strip()
  #############################################################
  
	for tok in nlp(sent):
		print(tok.text, "...", tok.dep_)

		# tok.text :: the actual word token
		# tok.dep_ :: the label (dependency tag)

		## chunk 2
		# if token is a punctuation mark then move on to the next token
		if tok.dep_ != "punct":
			# check: token is a compound word or not
			if tok.dep_ == "compound":
				prefix = tok.text
			# if the previous word was also a 'compound' then add the current word to it
			if prv_tok_dep == "compound":
				prefix = prv_tok_text + " "+ tok.text
			
			# check: token is a modifier or not
			if tok.dep_.endswith("mod") == True:
				if tok.text != "only":
					modifier = tok.text
					
			# if the previous word was also a 'compound' then add the current word to it
			if prv_tok_dep == "compound":
				modifier = prv_tok_text + " "+ tok.text
			
			if tok.dep_ == "prep":
				if not ent2 == "":
					ent2 = ent2 +" "+ tok.text
				else:
					ent1 = ent1 +" "+ tok.text
			## chunk 3
			if tok.dep_.find("subj") == True:
				ent1 = modifier +" "+ prefix + " "+ tok.text +" "+preposition
				prefix = ""
				modifier = ""
				prv_tok_dep = ""
				prv_tok_text = ""
				preposition = ""

			## chunk 4
			if tok.dep_.find("obj") == True:
				ent2 = modifier +" "+ prefix +" "+ tok.text



			## chunk 5  
			# update variables
			prv_tok_dep = tok.dep_
			prv_tok_text = tok.text
			# print("prv_tok_dep :: {}".format(prv_tok_dep))
			# print("prv_tok_text :: {}\n".format(prv_tok_text))
			# print("ent1 :: {}".format(ent1))
			# print("ent2 :: {}".format(ent2))
  #############################################################
	return [ent1.strip(), ent2.strip()]

def entity_extract_POS(sent):
	tokenise = word_tokenize(sent)
	pos_tagged = nltk.pos_tag(tokenise)

	for i in range(len(tokenise)):
		print(pos_tagged[i])

def get_relation(sent):

	doc = nlp(sent)

	# Matcher class object 
	matcher = Matcher(nlp.vocab)

	#define the pattern 
	pattern = [{'DEP':'ROOT'}, 
			{'DEP':'prep','OP':"?"},
			{'DEP':'agent','OP':"?"},  
			{'POS':'ADJ','OP':"?"}]

	matcher.add("matching_1", None, pattern) 

	matches = matcher(doc)
	
	k = len(matches) - 1

	span = doc[matches[k][1]:matches[k][2]] 

	return(span.text)

def testing(sentence):
    doc = nlp(sentence)

    for tok in doc:
        print(tok.text, "...", tok.dep_)

    print(get_entities(sentence))

def train_all_data():
    pd.set_option('display.max_colwidth', 200)

    # import data
    # candidate_sentences = pd.read_csv("../archive/all-data.csv", encoding='ISO-8859-1', usecols=['sentence'])
    # print(candidate_sentences.shape)
    # print(candidate_sentences['sentence'].sample(5))
    # pickle_out = open("../pickled_files/dataset.pickle","wb")
    # pickle.dump(candidate_sentences, pickle_out)
    # pickle_out.close()

    pickle_in = open("../pickled_files/dataset.pickle","rb")
    candidate_sentences = pickle.load(pickle_in)

    # extract entity pairs from each sentence
    # entity_pairs = []

    # for i in tqdm(candidate_sentences["sentence"]):
    #   entity_pairs.append(get_entities(i))

    # pickle_out = open("../pickled_files/entity_pairs.pickle","wb")
    # pickle.dump(entity_pairs, pickle_out)
    # pickle_out.close()

    pickle_in = open("../pickled_files/entity_pairs.pickle","rb")
    entity_pairs = pickle.load(pickle_in)

    # extract relations between the entities
    # relations = [get_relation(i) for i in tqdm(candidate_sentences['sentence' ])]
    # pickle_out = open("../pickled_files/relations.pickle","wb")
    # pickle.dump(relations, pickle_out)
    # pickle_out.close()

    pickle_in = open("../pickled_files/relations.pickle","rb")
    relations = pickle.load(pickle_in)

    ## building knowledge graph
    # extract subject
    source = [i[0] for i in entity_pairs]

    # extract object
    target = [i[1] for i in entity_pairs]

    kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})

    # create a directed-graph from a dataframe
    # G=nx.from_pandas_edgelist(kg_df, "source", "target", 
    #                           edge_attr=True, create_using=nx.MultiDiGraph())

    G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="increased by"], "source", "target", 
                              edge_attr=True, create_using=nx.MultiDiGraph())


    # plot the knowledge graph
    plt.figure(figsize=(12,12))

    pos = nx.spring_layout(G)
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.get_cmap("Blues"), pos = pos)
    plt.show()

def tf_idf(reviews):
	cv = CountVectorizer() 
	word_count_vector = cv.fit_transform(reviews)

	tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True) 
	tfidf_transformer.fit(word_count_vector)

	# print idf values 
	df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"]) 
	
	# sort ascending 
	df_idf.sort_values(by=['idf_weights'])
	# print(df_idf[0:20])
	# count matrix 
	count_vector = cv.transform(reviews) 
	
	# tf-idf scores 
	tf_idf_vector = tfidf_transformer.transform(count_vector)

	feature_names = cv.get_feature_names() 
	
	#get tfidf vector for first document 
	first_document_vector = tf_idf_vector[0] 
	
	#print the scores 
	df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"]) 
	print(df.sort_values(by=["tfidf"],ascending=False)[0:20])
	


def train_articles_data():
	pd.set_option('display.max_colwidth', 200)

	# import data
	# candidate_sentences = pd.read_csv("../selected.csv", encoding='ISO-8859-1', usecols=["Positive_Review"], nrows=5)

	hotel_reviews = pd.read_csv("../data/Hotel_Reviews.csv", usecols=["Hotel_Name", "Positive_Review"])
	candidate_sentences = hotel_reviews.loc[hotel_reviews['Hotel_Name'] == 'Hotel Arena']

	print(candidate_sentences.shape)
	# print(candidate_sentences['Positive_Review'].head(5))
  
	# pickle_out = open("../pickled_files/positive_reviews.pickle","wb")
	# pickle.dump(candidate_sentences, pickle_out)
	# pickle_out.close()

	# pickle_in = open("../pickled_files/positive_reviews.pickle","rb")
	# candidate_sentences = pickle.load(pickle_in)

	reviews = candidate_sentences["Positive_Review"]
	tf_idf(reviews)


	# extract entity pairs from each sentence
	# entity_pairs = []
	# relations = []
	# for i in tqdm(candidate_sentences["Positive_Review"]):
	# 	sentences = re.findall('[A-Z][^A-Z]*',i)
	# 	for sentence in sentences:
	# 		sentence = sentence.lower()
	# 		print("sentence = {}".format(sentence))

	# 		# entity_extracted = get_entities2(sentence)
	# 		# if entity_extracted[0] == '' or entity_extracted[1] == '':
	# 		# 	continue
	# 		# print("entity_extracted :: \n{}".format(entity_extracted))
	# 		# entity_pairs.append(entity_extracted)

	# 		entity_extract_POS(sentence)

	# 		relation_extracted = get_relation(sentence)
	# 		print("relation_extracted :: \n{}\n".format(relation_extracted))
	# 		relations.append(relation_extracted)


	# pickle_out = open("../pickled_files/positive_reviews_entity_pairs.pickle","wb")
	# pickle.dump(entity_pairs, pickle_out)
	# pickle_out.close()

	# pickle_in = open("../pickled_files/positive_reviews_entity_pairs.pickle","rb")
	# entity_pairs = pickle.load(pickle_in)

	# extract relations between the entities
	# relations = [get_relation(i) for i in tqdm(candidate_sentences["Positive_Review"])]
	# for i in 
	# pickle_out = open("../pickled_files/relations.pickle","wb")
	# pickle.dump(relations, pickle_out)
	# pickle_out.close()
	# print(pd.Series(relations).value_counts()[:50])

	# pickle_in = open("../pickled_files/relations.pickle","rb")
	# relations = pickle.load(pickle_in)
	# print(pd.Series(relations).value_counts()[:50])

	# for i in range(len(relations)):
	# 	print("{} -- {} -- {}".format(entity_pairs[i][0], relations[i], entity_pairs[i][1]))

	## building knowledge graph
	# extract subject
	# source = [i[0] for i in entity_pairs]

	# # extract object
	# target = [i[1] for i in entity_pairs]

	# kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})

	# create a directed-graph from a dataframe
	# G=nx.from_pandas_edgelist(kg_df, "source", "target", 
	#                           edge_attr=True, create_using=nx.MultiDiGraph())

	# G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="was amazing"], "source", "target", 
	# 							edge_attr=True, create_using=nx.MultiDiGraph())


	# # plot the knowledge graph
	# plt.figure("aqcuired",figsize=(12,12))
	# pos = nx.spring_layout(G, k=0.8)
	# nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.get_cmap("Blues"), pos = pos)
	# plt.show()


########
# Main #
########

train_articles_data()















'''
DEPENDENCIES tags list
LABEL	DESCRIPTION
acl	clausal modifier of noun (adjectival clause)
acomp	adjectival complement
advcl	adverbial clause modifier
advmod	adverbial modifier
agent	agent
amod	adjectival modifier
appos	appositional modifier
attr	attribute
aux	auxiliary
auxpass	auxiliary (passive)
case	case marking
cc	coordinating conjunction
ccomp	clausal complement
compound	compound
conj	conjunct
cop	copula
csubj	clausal subject
csubjpass	clausal subject (passive)
dative	dative
dep	unclassified dependent
det	determiner
dobj	direct object
expl	expletive
intj	interjection
mark	marker
meta	meta modifier
neg	negation modifier
nn	noun compound modifier
nounmod	modifier of nominal
npmod	noun phrase as adverbial modifier
nsubj	nominal subject
nsubjpass	nominal subject (passive)
nummod	numeric modifier
oprd	object predicate
obj	object
obl	oblique nominal
parataxis	parataxis
pcomp	complement of preposition
pobj	object of preposition
poss	possession modifier
preconj	pre-correlative conjunction
prep	prepositional modifier
prt	particle
punct	punctuation
quantmod	modifier of quantifier
relcl	relative clause modifier
root	root
xcomp	open clausal complement
'''

'''
POS tags list
TAG	 POS	MORPHOLOGY	DESCRIPTION
$	SYM		symbol, currency
``	PUNCT	PunctType=quot PunctSide=ini	opening quotation mark
''	PUNCT	PunctType=quot PunctSide=fin	closing quotation mark
,	PUNCT	PunctType=comm	punctuation mark, comma
-LRB-	PUNCT	PunctType=brck PunctSide=ini	left round bracket
-RRB-	PUNCT	PunctType=brck PunctSide=fin	right round bracket
.	PUNCT	PunctType=peri	punctuation mark, sentence closer
:	PUNCT		punctuation mark, colon or ellipsis
ADD	X		email
AFX	ADJ	Hyph=yes	affix
CC	CCONJ	ConjType=comp	conjunction, coordinating
CD	NUM	NumType=card	cardinal number
DT	DET		determiner
EX	PRON	AdvType=ex	existential there
FW	X	Foreign=yes	foreign word
GW	X		additional word in multi-word expression
HYPH	PUNCT	PunctType=dash	punctuation mark, hyphen
IN	ADP		conjunction, subordinating or preposition
JJ	ADJ	Degree=pos	adjective
JJR	ADJ	Degree=comp	adjective, comparative
JJS	ADJ	Degree=sup	adjective, superlative
LS	X	NumType=ord	list item marker
MD	VERB	VerbType=mod	verb, modal auxiliary
NFP	PUNCT		superfluous punctuation
NIL	X		missing tag
NN	NOUN	Number=sing	noun, singular or mass
NNP	PROPN	NounType=prop Number=sing	noun, proper singular
NNPS	PROPN	NounType=prop Number=plur	noun, proper plural
NNS	NOUN	Number=plur	noun, plural
PDT	DET		predeterminer
POS	PART	Poss=yes	possessive ending
PRP	PRON	PronType=prs	pronoun, personal
PRP$	DET	PronType=prs Poss=yes	pronoun, possessive
RB	ADV	Degree=pos	adverb
RBR	ADV	Degree=comp	adverb, comparative
RBS	ADV	Degree=sup	adverb, superlative
RP	ADP		adverb, particle
SP	SPACE		space
SYM	SYM		symbol
TO	PART	PartType=inf VerbForm=inf	infinitival “to”
UH	INTJ		interjection
VB	VERB	VerbForm=inf	verb, base form
VBD	VERB	VerbForm=fin Tense=past	verb, past tense
VBG	VERB	VerbForm=part Tense=pres Aspect=prog	verb, gerund or present participle
VBN	VERB	VerbForm=part Tense=past Aspect=perf	verb, past participle
VBP	VERB	VerbForm=fin Tense=pres	verb, non-3rd person singular present
VBZ	VERB	VerbForm=fin Tense=pres Number=sing Person=three	verb, 3rd person singular present
WDT	DET		wh-determiner
WP	PRON		wh-pronoun, personal
WP$	DET	Poss=yes	wh-pronoun, possessive
WRB	ADV		wh-adverb
XX	X		unknown
_SP	SPACE	

src = https://spacy.io/api/annotation
'''