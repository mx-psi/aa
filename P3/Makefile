all: trabajo3.zip

%.pdf: %.md
	pandoc -F pandoc-citeproc -o $@ $<

trabajo3.zip: memoria.pdf trabajo3.py
	zip -9 $@ -r $^
