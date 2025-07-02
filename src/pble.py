import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import  r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Carregar dados CSV, XLSX e Mapa
pble_path = 'databases/pble.csv'  # Caminho do arquivo
df_pble = pd.read_csv(pble_path, encoding='latin-1')  # Carregar a base

dtb_path = 'databases/RELATORIO_DTB_BRASIL_MUNICIPIO.xlsx'
df_dtb = pd.read_excel(dtb_path)

mun_path = 'databases/municipios.csv'
df_mun = pd.read_csv(mun_path, encoding='latin-1')

# Apresentação
st.title("Ciência de Dados")
st.write("Aluno: João Vitor Campõe Galescky")
st.write("Curso: 2° Tecnólogo em Análise e Desenvolvimento de Sistemas")

st.subheader("Programa Banda Larga nas Escolas")
st.write("O Programa Banda Larga na Escola (PBLE) foi lançado em 4 de abril de 2008 pelo Governo Federal, substituindo a obrigatoriedade das operadoras na instalação de Postos de Serviços Telefônicos (PST) pela infraestrutura de rede para suporte a conexão à Internet em alta velocidade para todos os municípios e escolas públicas urbanas.")

# Filtrar colunas relevantes
dfFilter = df_pble[["sigla_uf", "id_municipio", "rede","id_escola", "empresa", "tecnologia", "conexao"]]
dfFilter['conexao'] = dfFilter['conexao'].str.lower().str.replace(' mbps', '').str.replace(' Mbps', '').astype(float)

# Renomear coluna
df_dtb.rename(columns={"Codigo_Municipio" : "id_municipio"}, inplace=True)
df_mun.rename(columns={"codigo_ibge": "id_municipio"}, inplace=True)

dfFilter = pd.merge(dfFilter, df_dtb[['id_municipio', 'Nome_Municipio']], on="id_municipio", how="left")
dfFilter = pd.merge(dfFilter, df_mun[['id_municipio', 'latitude', 'longitude']], on="id_municipio", how="left")

# Seleção de estado
estado = st.multiselect("Escolha um estado:", dfFilter["sigla_uf"].unique())
if estado:
    dfUF = dfFilter[dfFilter['sigla_uf'].isin(estado)].drop(columns=['id_municipio'])
    st.dataframe(dfUF)
else:
    dfUF = dfFilter.drop(columns=['id_municipio'])
    st.dataframe(dfUF)
    
# Filtro por tecnologia ou empresa
opcao = st.radio("Filtrar tecnologia por:", ("UF", "Empresa"))
if opcao == "UF":
    tecn_estado = st.multiselect("Escolha uma tecnologia:", dfUF["tecnologia"].unique())
    if tecn_estado:
        dfSaida = dfUF[dfUF['tecnologia'].isin(tecn_estado)]
        st.dataframe(dfSaida)

elif opcao == "Empresa":
    empresa = st.multiselect("Escolha uma empresa:", dfUF["empresa"].unique())
    if empresa:
        dfEmpresa = dfUF[dfUF['empresa'].isin(empresa)]
        tecn_empresa = st.multiselect("Escolha uma tecnologia pela empresa:", dfEmpresa["tecnologia"].unique())
        if tecn_empresa:
            dfSaida = dfEmpresa[dfEmpresa['tecnologia'].isin(tecn_empresa)]
            st.dataframe(dfSaida)
            
# --- Gráfico 1: Distribuição de tecnologia ---
st.subheader("Distribuição de Tecnologia por Estado")
tecnologia_count = dfUF["tecnologia"].value_counts().reset_index()
tecnologia_count.columns = ['Tecnologia', 'Contagem']
fig1 = px.bar(tecnologia_count, x='Tecnologia', y='Contagem', title='Distribuição de Tecnologia', labels={'Contagem': 'Número de Escolas'})
st.plotly_chart(fig1)

# --- Gráfico 2: Entidades (Federal, Estadual, Municipal, Privada) ---
st.subheader("Distribuição de Redes por Entidade")
rede_count = dfUF["rede"].value_counts().reset_index()
rede_count.columns = ['Rede', 'Contagem']
fig2 = px.pie(rede_count, values='Contagem', names='Rede', title='Distribuição por Entidade (Federal, Estadual, Municipal, Privada)')
st.plotly_chart(fig2)

# --- Gráfico 3: Velocidade de Conexão por Tecnologia ---
st.subheader("Distribuição de Velocidade de Conexão")
velocidade_tecnologia = dfUF.groupby('tecnologia')['conexao'].mean().reset_index()
fig3 = px.bar(velocidade_tecnologia, x='tecnologia', y='conexao', title='Velocidade Média de Conexão por Tecnologia', labels={'tecnologia': 'Tecnologia', 'conexao': 'Velocidade (Mbps)'})
st.plotly_chart(fig3)

# --- Gráfico 4: Velocidade de Conexão por Estado ---
velocidade_estado = dfFilter.groupby('sigla_uf')['conexao'].mean().reset_index()
velocidade_estado['cor'] = velocidade_estado['sigla_uf'].apply(lambda x: 'highlighted' if x in estado else 'normal')
color_discrete_map = {'highlighted': '#511CFB', 'normal': '#19D3F3'}
fig_estado = px.bar(velocidade_estado, x='sigla_uf', y='conexao',title='Velocidade Média de Conexão por Estado', labels={'conexao': 'Velocidade (Mbps)', 'sigla_uf': 'UF'}, color='cor', color_discrete_map=color_discrete_map)
fig_estado.update_layout(showlegend=False)
st.plotly_chart(fig_estado)

# --- Gráfico 5: Velocidade de Conexão por Município ---
if estado:
    df_estado_municipio = dfFilter[dfFilter['sigla_uf'].isin(estado)]
else:
    df_estado_municipio = dfFilter

# Calcular a velocidade média de conexão
velocidade_municipio = df_estado_municipio.groupby('Nome_Municipio')['conexao'].mean().reset_index()

# Seleção
st.subheader("Destaque Municípios para Comparação")
cidade = df_estado_municipio['Nome_Municipio'].unique()
selecionado = st.multiselect("Escolha até 10 municípios:", cidade, max_selections=10)

if selecionado:
    velocidade_municipio = velocidade_municipio[velocidade_municipio['Nome_Municipio'].isin(selecionado)]
    velocidade_municipio['cor'] = 'normal'
    velocidade_municipio.loc[velocidade_municipio['Nome_Municipio'] == selecionado[0], 'cor'] = 'highlighted'
else:
    velocidade_municipio['cor'] = 'normal'
    
color_discrete_map = {
    'highlighted': '#511CFB',
    'normal': '#19D3F3'
}

# Plotar o gráfico
fig_municipio = px.bar(
    velocidade_municipio,
    x='Nome_Municipio',
    y='conexao',
    title='Velocidade Média de Conexão por Município',
    labels={'conexao': 'Velocidade (Mbps)', 'Nome_Municipio': 'Município'},
    color='cor',
    color_discrete_map=color_discrete_map
)
fig_municipio.update_layout(showlegend=False)
st.plotly_chart(fig_municipio)

# --- Gráfico 6: Velocidade de Conexão por Entidade  ---
df_entidades = dfFilter[dfFilter['rede'].isin(['Estadual', 'Municipal', 'Federal', 'Privada'])]
velocidade_entidade = df_entidades.groupby('rede')['conexao'].mean().reset_index()
fig_entidades = px.bar(velocidade_entidade, x='rede', y='conexao', title='Velocidade Média de Conexão por Entidade', labels={'conexao': 'Velocidade (Mbps)', 'rede': 'Entidade'})
st.plotly_chart(fig_entidades)

# --- Mapa Interativo por Estado Selecionado ---
st.title("Mapa Interativo de Conexão de Internet por Município")

if estado:
    dfUF = dfFilter[dfFilter['sigla_uf'].isin(estado)].drop(columns=['id_municipio'])
else:
    dfUF = dfFilter.drop(columns=['id_municipio'])

# Remover registros sem coordenadas para evitar erros no mapa
dfUF = dfUF.dropna(subset=['latitude', 'longitude'])

st.map(dfUF[['latitude', 'longitude']])

st.subheader("Dados de Conexão por Município")
st.dataframe(dfUF[['Nome_Municipio', 'sigla_uf', 'tecnologia', 'conexao', 'latitude', 'longitude']])

# --- Regressão Linear ---
df_pble['conexao'] = df_pble['conexao'].str.lower().str.replace(' mbps', '', regex=False).astype(float)
df_pble = df_pble[['tecnologia', 'conexao']].dropna()

st.subheader("Regressão Linear - Velocidade de Conexão por Tecnologia")

# One-Hot Encoding
encoder = OneHotEncoder()
encoded_tech = encoder.fit_transform(df_pble[['tecnologia']]).toarray()
encoded_tech_df = pd.DataFrame(encoded_tech, columns=encoder.get_feature_names_out(['tecnologia']))
df_encoded = pd.concat([df_pble[['conexao']], encoded_tech_df], axis=1)

# Dividir dados em treino e teste
x = df_encoded.drop(columns=['conexao'])
y = df_encoded['conexao']
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)

# Treinar modelo
model = LinearRegression()
model.fit(x_treino, y_treino)

# Previsões
y_pred = model.predict(x_teste)

# Avaliar o modelo
mse = mean_squared_error(y_teste, y_pred)
r2 = r2_score(y_teste, y_pred)

st.markdown(f"Erro Médio Quadrático (MSE): {mse:.2f}\n\nCoeficiente de Determinação (R²): {r2:.2f}")

# Coeficientes da regressão
coeficientes = pd.DataFrame({
    'Tecnologia': encoder.get_feature_names_out(['tecnologia']),
    'Coeficiente': model.coef_
}).sort_values(by='Coeficiente', ascending=False)

# Gráfico: comparação entre valores reais e previstos
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_teste, y_pred, alpha=0.7, color='blue')
ax.plot([y_teste.min(), y_teste.max()], [y_teste.min(), y_teste.max()], color='red', linestyle='--')
ax.set_title("Valores Reais vs. Previstos - Regressão Linear")
ax.set_xlabel("Velocidade Real (Mbps)")
ax.set_ylabel("Velocidade Prevista (Mbps)")

# Plotly para interação
fig_reg = px.scatter(
    x=y_teste,
    y=y_pred,
    labels={'x': 'Velocidade Real (Mbps)', 'y': 'Velocidade Prevista (Mbps)'},
    title="Valores Reais vs. Previstos - Interativo"
)
fig_reg.add_shape(type="line", x0=y_teste.min(), x1=y_teste.max(), y0=y_teste.min(), y1=y_teste.max(),
                  line=dict(color="red", dash="dash"))
st.plotly_chart(fig_reg)

# --- Regressão Logística ---
# Definir o limite para classificação binária (ex.: velocidade > 10 Mbps é 1, caso contrário 0)
limite = st.slider("Defina o limite para classificar velocidade alta (Mbps):", min_value=1, max_value=100, value=10)
df_pble['velocidade_alta'] = np.where(df_pble['conexao'] > limite, 1, 0)

st.subheader("Análise de Regressão Logística: Classificação de Velocidade")
st.write("Prevê se a velocidade de conexão ultrapassa um limite, com base no tipo de tecnologia.")

# One-Hot Encoding
encoder = OneHotEncoder()
encoded_tech = encoder.fit_transform(df_pble[['tecnologia']]).toarray()
encoded_tech_df = pd.DataFrame(encoded_tech, columns=encoder.get_feature_names_out(['tecnologia']))
df_encoded = pd.concat([df_pble[['velocidade_alta']], encoded_tech_df], axis=1)

# Dividir os dados em treino e teste
x = df_encoded.drop(columns=['velocidade_alta'])  # Variáveis independentes (tecnologia)
y = df_encoded['velocidade_alta']  # Variável dependente (velocidade alta ou não)
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)

# Treinar modelo
model = LogisticRegression()
model.fit(x_treino, y_treino)

# Previsões
y_pred = model.predict(x_teste)

st.subheader("Resultados do Modelo")
# st.write("Relatório de Classificação:")
# st.text(classification_report(y_teste, y_pred))

# Matriz de Confusão
st.write("Matriz de Confusão:")
conf_matrix = confusion_matrix(y_teste, y_pred)
st.write(pd.DataFrame(conf_matrix, columns=["Pred. Não", "Pred. Sim"], index=["Real Não", "Real Sim"]))

# Coeficientes do modelo
coefficients = pd.DataFrame({
    'Tecnologia': encoder.get_feature_names_out(['tecnologia']),
    'Coeficiente': model.coef_[0]
}).sort_values(by='Coeficiente', ascending=False)
st.write("Impacto das Tecnologias na Probabilidade de Alta Velocidade:")
st.dataframe(coefficients)

# Gráfico: Probabilidades previstas
y_prob = model.predict_proba(x_teste)[:, 1]
df_prob = pd.DataFrame({'Velocidade Real': y_teste, 'Probabilidade Prevista': y_prob})
fig = px.scatter(df_prob, x='Velocidade Real', y='Probabilidade Prevista', title="Probabilidades Previstas vs. Realidade")
st.plotly_chart(fig)

# --- Random Forest ---
st.subheader("Random Forest - Classificação de Velocidade Alta")

# Treinamento do modelo
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_treino, y_treino)

# Previsões
rf_y_pred = rf_model.predict(x_teste)
rf_y_prob = rf_model.predict_proba(x_teste)[:, 1]

# Avaliação do modelo
# st.write("Relatório de Classificação:")
# st.text(classification_report(y_teste, rf_y_pred))

# Matriz de Confusão
st.write("Matriz de Confusão:")
rf_conf_matrix = confusion_matrix(y_teste, rf_y_pred)
st.write(pd.DataFrame(rf_conf_matrix, columns=["Pred. Não", "Pred. Sim"], index=["Real Não", "Real Sim"]))

rf_feature_importances = pd.DataFrame({
    'Feature': x.columns,
    'Importância': rf_model.feature_importances_
}).sort_values(by='Importância', ascending=False)
st.write("Importância das Features no Random Forest:")
st.dataframe(rf_feature_importances)

# Gráfico de probabilidades previstas
df_rf_prob = pd.DataFrame({'Velocidade Real': y_teste, 'Probabilidade Prevista': rf_y_prob})
fig_rf = px.scatter(
    df_rf_prob, x='Velocidade Real',
    y='Probabilidade Prevista', 
    title="Random Forest: Probabilidades Previstas vs. Realidade"
)
st.plotly_chart(fig_rf)