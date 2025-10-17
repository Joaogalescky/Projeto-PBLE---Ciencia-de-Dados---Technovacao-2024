# Projeto - Analise de Dados do PBLE - Evento do Technovação

## Sobre o projeto
O projeto de Streamlit é um trabalho de estudo da ferramenta do Streamlit na matéria de Ciência de Dados, desenvolvida durante a 3° bimestre do 2° ano de Tecnologia em Análise e Desenvolvimento de Sistemas no ano de 2024.

Este trabalho visa **analisar** o **nível de conexão** das escolas no estado do Paraná, a **rede à qual pertence**, o nível de **abrangência do serviço**, as **redes que são atendidas**, o **volume de atendimento** e a **qualidade de conexão** fornecida, conforme os dados disponibilizados pela Agência Nacional de Telecomunicações **(ANATEL)**, sobre o Programa Banda Larga nas Escolas **(PBLE)**, tendo como tempo de cobertura atemporal do programa no ano de 2008, até o ano de 2021, que abrange escolas das esferas municipal, estadual, federal e particular. 

O projeto foi desenvolvido em acompanhamento nas aulas do prof. Darlon Vasata, utilizando-se de bases de dados, sendo um arquivo do tipo **CSV** e **XLSX**, disponíveis em: [Base dos Dados](https://basedosdados.org/dataset/4ba41417-ba19-4022-bc24-6837db973009?table=62e2ad04-2b2e-42aa-909d-9f44a819547e), proveniente da ANATEL e a [Divisão Territorial Brasileira **(DTB)**](https://www.ibge.gov.br/geociencias/organizacao-do-territorio/estrutura-territorial/23701-divisao-territorial-brasileira.html) - proveniente do Instituto Brasileiro De Geografia e Estatística **(IBGE)**.

O banco de dados utilizada baseia-se no programa do Governo Federal, o PBLE, lançado em 4 de abril de 2008, que substitui a obrigatoriedade das operadoras na instalação de Postos de Serviços Telefônicos **(PST)** pela infraestrutura de rede para conexão à Internet há todos os municípios e escolas públicas, e tem como finalidade melhorar o ensino pela disponibilização de acesso a internet em todas as escolas.

As ferramentas utilizadas para o desenvolvimento deste trabalho foram a linguagem de programação Python, suas bibliotecas e do Streamlit para visualização dos dataframes, gráficos, colunas e informações da análise.

## Streamlit
É uma biblioteca open-source em Python que permite a criação de aplicativos web para análise de dados de forma extremamente rápida. Com ela, pode transformar scripts de dados em web apps compartilháveis com poucas linhas de código, podendo visualizar e interagir com seus dados de maneira dinâmica.

Se destaca por simplificar o desenvolvimento de interfaces interativas e dashboards, sendo ideal para prototipagem rápida e compartilhamento de dados.

**Principais vantagens estão:**

* Criação de interfaces interativas com widgets;
* Integração fácil com bibliotecas como Matplotlib, Seaborn e Plotly para visualização de dados;
* Facilidade para hospedar e compartilhar aplicativos na web;
* Ideal para machine learning, permitindo testar modelos, ajustar hiperparâmetros e visualizar previsões em tempo real;
* Inclusão de widgets como sliders, checkboxes e seletores, facilitando a interação do usuário com dados e simulações.

## Bibliotecas utilizadas
* streamlit;
* numpy;
* pandas;
* matplotlib;
* plotly;
* openpyxl;
* geopandas;
* seaborn.

## Instruções
Para fazer a instalação de todos as bibliotecas necessárias para o funcionamento do projeto, pode-rá utilizar o **Makefile** ou o comando **pip**:

### Comando Makefile
```bash
make install
```

### Comando Python
```bash
pip3 install -r requirements.txt
```

### Inicialização
Entre no diretório **src** ou em **Trab4B** e execute o seguinte comando:
```bash
streamlit run pble.py
```

## Referências
https://hub.asimov.academy/blog/streamlit-guia-completo/

https://docs.streamlit.io/develop/api-reference/charts/st.map

https://hub.asimov.academy/tutorial/exibindo-mapas-com-streamlit-de-maneira-simples/

https://github.com/kelvins/municipios-brasileiros/blob/main/csv

https://github.com/alanwillms/geoinfo/blob/master/latitude-longitude-cidades.csv

## Autores
**Aluno:** João Vitor Campõe Galescky

**Orientador:** Darlon Vasata

**Co-orientador:** Edmar André Bellorini

# Instituição

[![IFPR Logo](https://user-images.githubusercontent.com/126702799/234438114-4db30796-20ad-4bec-b118-246ebbe9de63.png)](https://www.ifpr.edu.br)

**Instituto Federal do Paraná - IFPR - Campus [Cascavel](https://ifpr.edu.br/cascavel/)**  
Curso: Tecnologia em Análise e Desenvolvimento de Sistemas.

---

> Documento elaborado com [StackEdit](https://stackedit.io).
