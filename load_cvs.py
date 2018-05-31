import PyPDF2

def get_pdf_text(cv_file):
    """Extracts text from a single PDF"""
    pdf_content = ''
    f = open(cv_file, 'rb')
    cv = PyPDF2.PdfFileReader(f)
    n_pages = cv.getNumPages()
    for n in range(n_pages):
        page = cv.getPage(n)
        page_content = page.extractText()
        pdf_content += page_content

    return pdf_content

