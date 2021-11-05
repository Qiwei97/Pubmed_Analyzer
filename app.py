import streamlit as st
from transformers import pipeline
import pandas as pd
import numpy as np
from questiongenerator import QuestionGenerator, print_qa
from annotated_text import annotated_text
import requests


#-----------------------------------------

@st.cache(allow_output_mutation=True)
def analyze_text(text, url="https://bern.korea.ac.kr/plain"):
    return requests.post(url, data={'sample_text': text}).json()

@st.cache(allow_output_mutation=True)
def load_models():

    # Load all the models
    summarizer = pipeline("summarization",
                          model='patrickvonplaten/led-large-16384-pubmed')

    QnA = pipeline("question-answering",
                   model='franklu/pubmed_bert_squadv2')

    qg = QuestionGenerator()

    return summarizer, QnA, qg

summarizer, QnA, qg = load_models()

#-----------------------------------------

# sidebar
st.sidebar.header('Welcome to the Pubmed Analyzer!')
nav = st.sidebar.selectbox('Navigation', ['Summarization', 'Analyze Text'])

# st.markdown(
#     """
#     <style>
#     section[data-testid="stSidebar"] div[class="css-1lcbmhc e1fqkh3o0"] {
#         background-image: linear-gradient(#8993ab,#8993ab);
#         color: white
#         }
#     </style>
#     """, unsafe_allow_html=True
# )

#-----------------------------------------

# Summarization
def p_title(title):
    st.markdown(f'<h3 style="text-align: left; color:#F63366; font-size:28px;">{title}</h3>', unsafe_allow_html=True)

if nav == 'Summarization':
    
    st.markdown("<h3 style='text-align: center; color:grey;'>Pubmed Analyzer &#129302;</h3>", unsafe_allow_html=True)
    st.text('')
    p_title('Summarization')
    st.text('')

    example = "Autophagy maintains tumour growth through circulating arginine. Autophagy captures intracellular components and delivers them to lysosomes, where they are degraded and recycled to sustain metabolism and to enable survival during starvation1-5. Acute, whole-body deletion of the essential autophagy gene Atg7 in adult mice causes a systemic metabolic defect that manifests as starvation intolerance and gradual loss of white adipose tissue, liver glycogen and muscle mass1. Cancer cells also benefit from autophagy. Deletion of essential autophagy genes impairs the metabolism, proliferation, survival and malignancy of spontaneous tumours in models of autochthonous cancer6,7. Acute, systemic deletion of Atg7 or acute, systemic expression of a dominant-negative ATG4b in mice induces greater regression of KRAS-driven cancers than does tumour-specific autophagy deletion, which suggests that host autophagy promotes tumour growth1,8. Here we show that host-specific deletion of Atg7 impairs the growth of multiple allografted tumours, although not all tumour lines were sensitive to host autophagy status. Loss of autophagy in the host was associated with a reduction in circulating arginine, and the sensitive tumour cell lines were arginine auxotrophs owing to the lack of expression of the enzyme argininosuccinate synthase 1. Serum proteomic analysis identified the arginine-degrading enzyme arginase I (ARG1) in the circulation of Atg7-deficient hosts, and in vivo arginine metabolic tracing demonstrated that serum arginine was degraded to ornithine. ARG1 is predominantly expressed in the liver and can be released from hepatocytes into the circulation. Liver-specific deletion of Atg7 produced circulating ARG1, and reduced both serum arginine and tumour growth."
        
    text = st.text_area("Use the example below or input your own text in English (between 1,000 and 10,000 characters)",
                        value=example, max_chars=10000, height=330)
    
    st.session_state.text = text
    st.session_state.qn_list = ""

    if st.button('Summarize'):
        if len(text) < 1000:
            st.error('Please enter a text in English of minimum 1,000 characters')
        else:
            with st.spinner('Processing...'):
                summary = summarizer(text, 
                                     min_length=50, 
                                     max_length=200,
                                     num_beams=3, 
                                     no_repeat_ngram_size=2, 
                                     early_stopping=True,
                                     clean_up_tokenization_spaces=True)[0]['summary_text']
                # Clean output
                summary = summary.split('.')
                summary = [i.strip().capitalize() for i in summary]
                summary = '. '.join(summary)[:-1]

                sum_stats = (str(len(summary.split())) + ' words' + ' ('"{:.0%}".format(len(summary.split())/len(text.split())) + ' of original content)')
                st.markdown('___')
                st.caption(sum_stats)
                st.success(summary)
                st.session_state.summary = summary

#-----------------------------------------

# QnA + NER
if nav == 'Analyze Text':
    
    # NER
    try:
        st.markdown("<h3 style='text-align: center; color:grey;'>Pubmed Analyzer &#129302;</h3>", unsafe_allow_html=True)
        st.text('')
        p_title('Named Entities')

        entities = analyze_text(st.session_state.summary)
        ent_df = pd.DataFrame(entities['denotations'])[['obj', 'span']]
        ent_df['Phrase'] = ent_df.span.apply(lambda x: st.session_state.summary[x['begin']:x['end']])
        ent_df['begin'] = ent_df.loc[:,'span'].apply(lambda x: x['begin'])

        no_ent = pd.DataFrame(ent_df.span.tolist())
        no_ent['lag'] = no_ent.end.shift()
        no_ent = no_ent.fillna(0)
        no_ent.lag = no_ent.lag.astype(int)
        no_ent['Phrase'] = no_ent.apply(lambda x: st.session_state.summary[x['lag']:x['begin']], axis = 1)
        no_ent['begin'] = no_ent.lag
        no_ent = no_ent[['Phrase', 'begin']]

        ent_df = ent_df.append(no_ent)
        ent_df = ent_df.sort_values('begin').drop(columns=['span'])
        ent_df = ent_df.fillna(0).reset_index(drop=True)

        ner_text = []
        for row in range(len(ent_df)):
            phrase = ent_df['Phrase'][row] + " "
            if ent_df['obj'][row] == 0:
                ner_text.append((phrase))
            elif ent_df['obj'][row] == 'disease':
                ner_text.append((phrase, ent_df['obj'][row], "#8ef"))
            elif ent_df['obj'][row] == 'drug':
                ner_text.append((phrase, ent_df['obj'][row], "#faa"))
            elif ent_df['obj'][row] == 'gene':
                ner_text.append((phrase, ent_df['obj'][row], "#fea"))
            else:
                ner_text.append((phrase, ent_df['obj'][row], "#afa"))
        
        ner_text.append(('.'))
        annotated_text(*ner_text)

    except:
        st.error('Please summarize your text first.')

    # QnA
    if 'summary' in st.session_state:
        st.markdown('___')
        p_title('Question Answering')
        question = st.text_input('Type your question here')
        if st.button('Submit'):
            with st.spinner('Processing...'):
                if question:
                    answer = QnA(question=question, context=st.session_state.text)['answer']
                    st.success(answer)
                else:
                    st.error("Please ask a question.")
    
    # QG
    if 'summary' in st.session_state:
        if st.session_state.qn_list == "":
            with st.spinner('Generating Suggestions...'):
                text = st.session_state.text.split('.')
                text = [i for i in text if i != ""]
                text = [i + '.' for i in text]
                text = np.random.choice(text, 5, replace=False)

                st.session_state.qn_list = []
                for i in text:
                    qn = qg.generate(i, num_questions=1, answer_style='all', use_evaluator=False)[0]['question']
                    st.session_state.qn_list.append(qn)

        st.text('')
        st.markdown("<h6 style=color:grey;'>Suggestions</h6>", unsafe_allow_html=True)
        for i in st.session_state.qn_list:
            st.caption(i.capitalize())
