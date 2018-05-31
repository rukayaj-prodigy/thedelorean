import glob
from load_cvs import get_pdf_text, is_proper_cv, extract_words
import pdb
from cv_images import pdf_contains_image
from extract_contact_details import get_address, get_email, get_phone_number
import PyPDF2
import enchant
import csv

cv_dir = '/Users/rukayajohaadien/Dropbox (Prodigy)/The Deloreans/Careers Data (CVs)/Good CVs'
cv_list = glob.glob('{}/*.pdf'.format(cv_dir))

desirable_features = {
    ''
}

d = enchant.Dict("en_US")

bluechip_companies = open('bluechip_companies.csv', 'rb').read().splitlines()

with open('output.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['File name', 'Has image?', 'Is too long?', 'Email', 'Phone', 'Address', 'Spelling rating', 'Bluechip rating'])

    for cv_file in cv_list:
        # Load the cv
        f = open(cv_file, 'rb')
        cv = PyPDF2.PdfFileReader(f)

        has_image = pdf_contains_image(cv)
        is_too_long = cv.getNumPages() > 2

        cv_text = get_pdf_text(cv)

        if not is_proper_cv(cv_text):
            continue

        email = get_email(cv_text)
        phone = get_phone_number(cv_text)
        address = get_address(cv_text)

        badly_spelled = 0
        all_words = extract_words(cv_text)
        for word in all_words.split(' '):
            if word.strip():
                if not d.check(word):
                    badly_spelled += 1
        spelling_rating = round((1 - (badly_spelled/len(all_words.split(' '))))*100)

        bluechip_rating = 0
        for company in bluechip_companies:
            if str(company.rstrip(b','), 'utf-8') in all_words:
                bluechip_rating += 1

        csv_writer.writerow([cv_file.split('/')[-1], has_image, is_too_long, email, phone, address, spelling_rating, bluechip_rating])

