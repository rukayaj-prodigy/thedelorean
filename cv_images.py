def pdf_contains_image(cv):
    "Checks to see if a PDF contains an image."
    contains_image = False

    for pi in range(cv.numPages):
        pg = cv.getPage(pi)

        if '/XObject' in pg['/Resources']:
            xObject = pg['/Resources']['/XObject'].getObject()

            for obj in xObject:
                if xObject[obj]['/Subtype'] == '/Image':
                    contains_image = True
                    return contains_image
    return contains_image
