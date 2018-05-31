import re
import pyap

def get_email(cv_text):
    email = re.findall(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)",cv_text)
    return email


def get_phone_number(cv_text):
    # We're assuming the first phone number this finds in the CV is the one which belongs to this person
    phone_number = re.findall(r"(\d{3}[-\.\s]\d{3}[-\.\s]\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]\d{4}|\d{3}[-\.\s]\d{4})", cv_text)
    return phone_number


def get_address(cv_text):
    address = pyap.parse(cv_text.replace('\n', ''), country='US')
    return address
