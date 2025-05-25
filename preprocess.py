def preprocess_text(text):
    import re
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # hapus link
    text = re.sub(r'[^\w\s]', '', text)  # hapus tanda baca
    text = re.sub(r'\d+', '', text)       # hapus angka
    text = stemmer.stem(text)

    return text