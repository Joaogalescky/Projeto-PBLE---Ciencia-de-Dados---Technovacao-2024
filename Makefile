aula:
	@echo "Aula de Ciência de Dados"

all: run

install:
	pip3 install -r requirements.txt
	
run:
	streamlit run src/pble.py

git:
	git config --global user.email "joaocvgalescky@gmail.com"
	git config --global user.name  "Joaogalescky"