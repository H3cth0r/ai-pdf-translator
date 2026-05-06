import fitz
path = "../oulu/jobs/vml.pdf"  # your PDF
doc = fitz.open(path)
for i in range(len(doc)):
    print(f"--- page {i} ---")
    for f in doc[i].get_fonts():  # (xref, ext, type, basefont, name, ref)
        print(f)
doc.close()