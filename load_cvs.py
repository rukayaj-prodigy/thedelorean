import PyPDF2
import re


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


def extract_words(cv_text):
    new_text = re.sub(r'[^\w ]', '', cv_text)
    return re.sub(r'\s+', ' ', new_text)

