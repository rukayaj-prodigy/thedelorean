def pdf_contains_image(cv):
    "Checks to see if a PDF contains an image."

    for pi in range(cv.numPages):
        pg = cv.getPage(pi)

        if '/XObject' in pg['/Resources']:
            xObject = pg['/Resources']['/XObject'].getObject()

            for obj in xObject:
                if xObject[obj]['/Subtype'] == '/Image':
                    # If it's in portrait it's a headshot (our assumption)
                    if xObject[obj]['/Height'] > xObject[obj]['/Width']:
                        return True
    return False
