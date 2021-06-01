import re
from joblib import load
import pandas as pd
import spacy
from PIL import Image

import spacy.cli 
spacy.cli.download("de_core_news_sm")

import streamlit as st


st.title('Tone analysis of smartphone reviews on German')

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://www.myclickmagazine.com/wp-content/uploads/2019/02/Photographing_Minimalism_Composition_Negative_Space_by_Dana_Walton_18.jpg")
    }
    .font-1 {
    font-size:30px !important;
    color: darkblue;
    }

    .font-2 {
    font-size:30px !important;
    color: forestgreen;
    }

    .font-3 {
    font-size:20px !important;
    color: black;
    }
    """,
    unsafe_allow_html=True
)

model = load('model.joblib')

punct_list = '["#$%&\'()*+,-./:;<=>@[\]^_`{|}~’‘´`\']'
nlp = spacy.load('de_core_news_sm')
d = {0:'negative review.', 1:'positive review.'}

option = st.text_input('Type in your review on smartphone (only German language supported)')

X_test = option

s = 'Review:\t' + X_test
st.markdown(f'<p class="font-1">{s}</p>', unsafe_allow_html=True)

if X_test:
    
    X_test = X_test.lower()
    X_test = re.sub(punct_list, "", X_test)
    X_test = re.sub("\d+", "", X_test)
    X_test = " ".join([token.lemma_ for token in nlp(X_test)])

    pred = model.predict([X_test])
    pred = d[pred[0]]
    confidence = abs(model.predict_proba([X_test])[0][0] - 0.5)

    if confidence < 0.1:
        confidence = 'Maybe it is'
    elif confidence > 0.4:
        confidence = 'It is almost sure that it is'
    else:
        confidence = 'It is possible that it is'
        
    s = "Conclusion:\t" + " ".join([confidence, pred])

    st.markdown(f'<p class="font-2">{s}</p>', unsafe_allow_html=True)

top = pd.read_csv('top.csv')

if st.checkbox('Show top-20 collocations for positive and negative reviews'):
    top

if st.checkbox('Show visual analysis of brands'):
    for i in ['Price_dist_brand.png', 'Rate_dist_brands.png', 'review_dist.png']:
        image = Image.open(i)
        st.image(image)

if st.checkbox('Show the comparison between rates of Apple and Samsung'):
    image = Image.open('samsung_apple.png')
    st.image(image)
    s = 'Using Mann-Whitney\'s Test, we can establish that rates of user satisfaction are rather different for Apple and Samsung.'
    st.markdown(f'<p class="font-3">{s}</p>', unsafe_allow_html=True)
