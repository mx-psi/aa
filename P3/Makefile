all: trabajo3.zip

%.pdf: %.md
	pandoc -o $@ $<

trabajo3.zip: memoria.pdf trabajo3.py
	zip -9 $@ -r $^
