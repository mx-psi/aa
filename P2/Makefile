all: trabajo2.zip

%.pdf: %.md
	pandoc -o $@ $<

trabajo2.zip: memoria.pdf trabajo2.py
	zip -9 $@ -r $^
