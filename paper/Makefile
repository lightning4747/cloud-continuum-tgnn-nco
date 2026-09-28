# Makefile for IEEE Conference Paper

MAIN = main
LATEX = pdflatex
BIBTEX = bibtex

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib IEEEtran.cls figures/*
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.synctex.gz

distclean: clean
	rm -f $(MAIN).pdf

.PHONY: all clean distclean
