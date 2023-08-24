from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet
import bs4 as bs
import warnings
import urllib.request
import nltk
import random
import string
import re

nltk.download('wordnet')
nltk.download('punkt')

warnings.filterwarnings('ignore')

synonyms = []
for syn in wordnet.synsets('hello'):
    for lem in syn.lemmas():
        lem_name = re.sub(r'\[[0-9]*\]', ' ', lem.name())
        lem_name = re.sub(r'\s+', ' ', lem.name())
        synonyms.append(lem_name)

greeting_inputs = ['hey', 'whats up', 'good morning', 'good evening', 'morning', 'evening', 'hello there', 'hey there']

greeting_inputs = greeting_inputs + synonyms

covo_inputs = ['how are you', 'how are you doing', 'you good']

greeting_responses = ['Hello! How can I help you?',
                      'Hey there! So what do you want to know?',
                      'Hi, you can ask me anything regarding ChatGpt2.',
                      'Hey! wanna know about ChatGpt2? Just ask away!']

convo_responses = ['Great! what about you?', 'Getting bored at home :( wbu??', 'Not too shabby']

convo_replies = ['great', 'i am fine', 'fine', 'good', 'super', 'superb', 'super great', 'nice']

question_answers = {'what are you': 'I am bot, ro-bot :3',
                    'who are you': 'I am bot, ro-bot :3',
                    'what can you do': 'Answer questions regarding ChatGpt2!',
                    'what do you do': 'Answer questions regarding ChatGpt2!'}


raw_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/ChatGPT')

raw_data = raw_data.read()

article = bs.BeautifulSoup(raw_data, 'lxml')

paragraphs = article.find_all('p')

article_text = ''

for p in paragraphs:
    article_text += p.text

article_text = article_text.lower()

article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
article_text = re.sub(r'\s+', ' ', article_text)

sentences = nltk.sent_tokenize(article_text)

words = nltk.word_tokenize(article_text)

lemma = nltk.stem.WordNetLemmatizer()

def perform_lemmatization(tokens):
    return [lemma.lemmatize(token) for token in tokens]

remove_punctuation = dict((ord(punc), None) for punc in string.punctuation)

def processed_data(document):
    return perform_lemmatization(nltk.word_tokenize(document.lower().translate(remove_punctuation)))

def punc_remove(str):
    punctuations = r'''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punct = ''

    for char in str:
        if char not in punctuations:
            no_punct = no_punct + char

    return no_punct

def generate_greeting_response(hello):
    if punc_remove(hello.lower()) in greeting_inputs:
        return random.choice(greeting_responses)

def generate_convo_response(str):
    if punc_remove(str.lower()) in covo_inputs:
        return random.choice(convo_responses)

def generate_answers(str):
    if punc_remove(str.lower()) in question_answers:
        return question_answers[punc_remove(str.lower())]

def generate_response(user):
    ChatGPT2 = ''
    sentences.append(user)

    word_vectorizer = TfidfVectorizer(tokenizer=processed_data, stop_words='english')
    all_word_vectors = word_vectorizer.fit_transform(sentences)
    similar_vector_values = cosine_similarity(all_word_vectors[-1], all_word_vectors)
    similar_sentence_number = similar_vector_values.argsort()[0][-2]

    matched_vector = similar_vector_values.flatten()
    matched_vector.sort()
    vector_matched = matched_vector[-2]

    if vector_matched is 0:
        ChatGPT2_response = ChatGPT2_response + 'Sorry, my database doesn\'t have the response for that. Try ' \
                                                'something different and related to ChatGpt2. '
        return ChatGPT2_response
    else:
        ChatGPT2_response = ChatGPT2_response + sentences[similar_sentence_number]
        return ChatGPT2_response


# chatting with the chatbot -->
continue_chat = True
print('Hi! I am ChatGpt2. How may I help you today... ')
while continue_chat:
    user_input = input().lower()
    user_input = punc_remove(user_input)
    if user_input != 'bye':
        if user_input == 'thanks' or user_input == 'thank you very much' or user_input == 'thank you':
            continue_chat = False
            print('ChatGpt2: Not a problem! (And WELCOME! :D)')
        elif user_input in convo_replies:
            print('That\'s nice! How may I be of assistance?')
            continue
        else:
            if generate_greeting_response(user_input) is not None:
                print('ChatGpt2: ' + generate_greeting_response(user_input))
            elif generate_convo_response(user_input) is not None:
                print('ChatGPT2: ' + generate_convo_response(user_input))
            elif generate_answers(user_input) is not None:
                print('ChatGPT2: ' + generate_answers(user_input))
            else:
                print('ChatGPT2: ', end='')
                print(generate_response(user_input))
                sentences.remove(user_input)
    else:
        continue_chat = False
        print('ChatGPT2: Bye, Have a nice day!')