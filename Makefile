TARG = understanding-parallel-scan

.PRECIOUS: %.tex %.pdf %.web

all: $(TARG).pdf

see: $(TARG).see

%.pdf: %.tex Makefile
	pdflatex $*.tex

%.tex: %.lhs macros.tex mine.fmt Makefile
	lhs2TeX -o $*.tex $*.lhs

showpdf = open -a Skim.app

%.see: %.pdf
	${showpdf} $*.pdf

clean:
	rm $(TARG).{tex,pdf,aux,nav,snm,ptb}


web: web-token

STASH=conal@conal.net:/home/conal/web/talks
web: web-token

web-token: $(TARG).pdf
	scp $? $(STASH)
	touch $@
