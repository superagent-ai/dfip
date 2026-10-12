.PHONY: analyze pdf clean
analyze:
	python ../scripts/analyze_paper.py
pdf: analyze
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
clean:
	latexmk -c main.tex
