
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Carregar o modelo treinado
modelo = joblib.load("modelo_randomForest.pkl")

# Carregar a base de treino para aplicar .fit() ao transformador
base_fit = pd.read_pickle("dados_limpos.pkl")
base_fit_features = base_fit.drop(columns='HiringDecision')

# Definir o transformador
coluna_transformada = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['Gender', 'RecruitmentStrategy']),
        ('num', StandardScaler(), [
            'Age', 'ExperienceYears', 'PreviousCompanies', 
            'DistanceFromCompany', 'InterviewScore', 
            'SkillScore', 'PersonalityScore'
        ]),
        ('educ', 'passthrough', ['EducationLevel'])
    ]
)

# Fazer fit com base nos dados originais
coluna_transformada.fit(base_fit_features)

# ========== Parte Individual ==========
def app_candidato_individual():
    st.title("🔍 Previsão para um Candidato")

    idade = st.number_input("Idade", 18, 60)
    genero = st.selectbox("Género", ["Masculino", "Feminino"])
    educacao = st.selectbox("Nível de Educação", ["Bachelor's (Tipo 1)", "Bachelor's (Tipo 2)", "Master's", "PhD"])
    experiencia = st.number_input("Anos de Experiência", 0, 40)
    empresas_previas = st.number_input("Nº de Empresas Anteriores", 0, 10)
    distancia = st.number_input("Distância até à Empresa (km)", 0.0, 100.0)
    entrevista = st.slider("Pontuação da Entrevista", 0, 100)
    tecnica = st.slider("Pontuação Técnica", 0, 100)
    personalidade = st.slider("Pontuação de Personalidade", 0, 100)
    estrategia = st.selectbox("Estratégia de Recrutamento", ["Agressiva", "Moderada", "Conservadora"])

    if st.button("Prever Contratação"):
        genero_map = {"Masculino": 0, "Feminino": 1}
        educ_map = {
            "Bachelor's (Tipo 1)": 1,
            "Bachelor's (Tipo 2)": 2,
            "Master's": 3,
            "PhD": 4
        }
        estrategia_map = {"Agressiva": 1, "Moderada": 2, "Conservadora": 3}

        input_dict = {
            'Age': idade,
            'Gender': genero_map[genero],
            'EducationLevel': educ_map[educacao],
            'ExperienceYears': experiencia,
            'PreviousCompanies': empresas_previas,
            'DistanceFromCompany': distancia,
            'InterviewScore': entrevista,
            'SkillScore': tecnica,
            'PersonalityScore': personalidade,
            'RecruitmentStrategy': estrategia_map[estrategia]
        }

        input_df = pd.DataFrame([input_dict])
        dados_transformados = coluna_transformada.transform(input_df)

        resultado = modelo.predict(dados_transformados)[0]
        st.subheader("Resultado da Previsão:")
        if resultado == 1:
            st.success("✅ Previsão: Contratar")
        else:
            st.warning("❌ Previsão: Não contratar")
            

# ========== Parte CSV ==========
def app_csv():
    st.title("📂 Previsão em Lote (CSV com dados)")
    ficheiro = st.file_uploader("Carregue um ficheiro CSV com os dados originais dos candidatos", type="csv")

    colunas_esperadas = [
        'Age', 'Gender', 'EducationLevel', 'ExperienceYears',
        'PreviousCompanies', 'DistanceFromCompany', 'InterviewScore',
        'SkillScore', 'PersonalityScore', 'RecruitmentStrategy'
    ]

    if ficheiro is not None:
        df = pd.read_csv(ficheiro)

        if all(col in df.columns for col in colunas_esperadas):
            st.success("✅ Ficheiro válido!")
            st.dataframe(df)

            dados_transformados = coluna_transformada.transform(df)
            df['Previsão'] = modelo.predict(dados_transformados)
            df['Previsão'] = df['Previsão'].apply(lambda x: 'Contratar' if x == 1 else 'Não contratar')

            st.subheader("📊 Resultados das Previsões")
            st.dataframe(df)

            st.subheader("📈 Distribuição das Previsões")
            contagem = df['Previsão'].value_counts()
            fig, ax = plt.subplots()
            contagem.plot(kind='bar', ax=ax, color=['tomato', 'seagreen'])
            ax.set_ylabel("Número de Candidatos")
            ax.set_xlabel("Resultado da Previsão")
            ax.set_title("Distribuição: Contratar vs Não Contratar")
            st.pyplot(fig)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV com Previsões", csv, "previsoes.csv", "text/csv")
        else:
            st.error("❌ O CSV deve conter as seguintes colunas:")
            st.code(colunas_esperadas)

# ======= Interface Principal ========
st.set_page_config(page_title="Previsão de Contratação", layout="centered")
pagina = st.sidebar.selectbox("Escolha a Página", ["Candidato Individual", "Lista de Candidatos"])

if pagina == "Candidato Individual":
    app_candidato_individual()
else:
    app_csv()
