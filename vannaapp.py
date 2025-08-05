# app.py
import streamlit as st
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore

# Classe combinando Ollama + ChromaDB
class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

# Cache do modelo
@st.cache_resource
def load_vanna():
    vn = MyVanna(config={'model': 'llama3'})
    vn.connect_to_mysql(
        host="localhost",
        dbname="cancerdb",
        user="root",
        password="admin123",
        port=3306
    )
    return vn

# Função de re-treino
def treinar_vanna(vn):
    df = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'cancerdb'")
    plano = vn.get_training_plan_generic(df)
    vn.train(plan=plano)

# ---- Interface ----
st.title("💊 Chat com dados de Pacientes de Câncer Global")
st.caption("Base: Classificação e características de pacientes de câncer")

vn = load_vanna()

# Botão de re-treino
if st.button("🔁 Re-treinar modelo"):
    with st.spinner("Re-treinando com os metadados da base..."):
        treinar_vanna(vn)
    st.success("✅ Re-treinamento concluído!")

# Campo de pergunta
pergunta = st.text_input("O que você gostaria se saber sobre pacientes de câncer globais:")

# Quando o usuário envia uma pergunta
if pergunta:
    with st.spinner("🔎 Processando pergunta..."):
        try:
            resposta = vn.ask(pergunta)
            sql = resposta[0] if isinstance(resposta, tuple) else resposta

            if sql is None or sql.strip() == "":
                st.error("❌ O modelo não conseguiu gerar uma consulta SQL válida.")
            else:
                st.markdown("#### 🧠 SQL gerado:")
                st.code(sql, language='sql')

                resultado = vn.run_sql(sql)

                st.markdown("#### 📋 Resultado da consulta:")
                if resultado is not None and len(resultado) > 0:
                    st.dataframe(resultado)
                else:
                    st.warning("⚠️ Consulta executada, mas nenhum dado foi retornado.")

        except Exception as e:
            st.error(f"❌ Erro ao processar a pergunta ou executar a SQL:\n\n{e}")
            
