import glob
from load_cvs import get_pdf_text
import pdb
import re

cv_dir = '/Users/rukayajohaadien/Dropbox (Prodigy)/The Deloreans/Careers Data (CVs)/Data Science 11Dec2017/20171211103034-IE-CVs'
cv_list = glob.glob('{}/*.pdf'.format(cv_dir))

for cv_file in cv_list:
    cv_text = get_pdf_text(cv_file)

    pdb.set_trace()


