import glob
from load_cvs import get_pdf_text, is_proper_cv
import pdb
from cv_images import pdf_contains_image
from extract_contact_details import get_address, get_email, get_phone_number
import PyPDF2

cv_dir = '/Users/rukayajohaadien/Dropbox (Prodigy)/The Deloreans/Careers Data (CVs)/Good CVs'
cv_list = glob.glob('{}/*.pdf'.format(cv_dir))

desirable_features = {
    ''
}


for cv_file in cv_list:
    # Load the cv
    f = open(cv_file, 'rb')
    cv = PyPDF2.PdfFileReader(f)

    has_image = pdf_contains_image(cv)
    is_too_long = cv.getNumPages() > 2

    cv_text = get_pdf_text(cv)

    is_proper_cv = is_proper_cv(cv_text)

    if not is_proper_cv:
        continue

    contact_details = {
        'email': get_email(cv_text),
        'phone': get_phone_number(cv_text),
        'address': get_address(cv_text)
    }


    pdb.set_trace()


