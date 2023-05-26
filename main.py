import streamlit as st
from joblib import load
import pandas as pd
#Uma aplicação de streamlit que mostra a execução de um modelo machine learning
st.markdown('# Verificador de estabilidade ⚡ ')

tau1 = st.number_input(label='tau1',value=9.857388)
tau2 = st.number_input(label='tau2',value=6.714277)
tau3 = st.number_input(label='tau3',value=4.545093)
tau4 = st.number_input(label='tau4',value=1.771546)	
p1 = st.number_input(label='p1',value=4.834561)
p2 = st.number_input(label='p2',value=-1.507867)
p3 = st.number_input(label='p3',value=-1.978492)
p4 = st.number_input(label='p4',value=-1.348202)
g1 = st.number_input(label='g1',value=0.775840)
g2 = st.number_input(label='g2',value=0.245026)
g3 = st.number_input(label='g3',value=0.604325)
g4 = st.number_input(label='g4',value=0.372714)

columns = ["tau1", "tau2", "tau3", "tau4", "p1", "p2", "p3", "p4", "g1", "g2", "g3", "g4"]
data = [[tau1, tau2, tau3, tau4, p1, p2, p3, p4, g1, g2, g3, g4]]
X_usuario = pd.DataFrame(data, columns=columns)
#Predições 
if st.button('Enviar'):
    model = load('modelo.pkl')
    final_pred = model.predict(X_usuario)[0]
    if final_pred[-1] == 'stable':
        st.success('### A rede está estável')
        st.balloons()
    else:
        st.error('### A rede está instável, verifique')