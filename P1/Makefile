all: trabajo1.zip

%.pdf: %.md
	pandoc -o $@ $<

trabajo1.zip: memoria.pdf trabajo1.py
	zip -9 $@ -r $^
