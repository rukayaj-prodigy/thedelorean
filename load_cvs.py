import PyPDF2


def is_proper_cv(cv_text):
    if len(cv_text) < 100:
        return False
    if cv_text.count('`') > 10:
        return False
    if cv_text.count('#') > 10:
        return False

    return True


def get_pdf_text(cv):
    """Extracts text from a single PDF"""
    pdf_content = ''
    n_pages = cv.getNumPages()
    for n in range(n_pages):
        page = cv.getPage(n)
        page_content = page.extractText()
        pdf_content += page_content

    return pdf_content

