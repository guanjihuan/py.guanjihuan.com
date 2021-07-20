# GUAN is an open-source python package developed and maintained by https://www.guanjihuan.com. The primary location of this package is on website https://py.guanjihuan.com.

# audio

def txt_to_audio(txt_path, rate=125, voice_type_0_or_1=1, read=1, save=0, print_text=0):
    import pyttsx3
    f = open(txt_path, 'r', encoding ='utf-8')
    text = f.read()
    if print_text==1:
        print(text)
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')  
    engine.setProperty('voice', voices[voice_type_0_or_1].id)
    engine.setProperty("rate", rate)
    if save==1:
        import re
        file_name = re.split('[/,\\\]', txt_path)[-1][:-4]
        engine.save_to_file(text, file_name+'.mp3')
        engine.runAndWait()
        print('MP3 file saved!')
    if read==1:
        engine.say(text)
        engine.runAndWait()

def pdf_to_text(pdf_path):
    from pdfminer.pdfparser import PDFParser, PDFDocument
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.converter import PDFPageAggregator
    from pdfminer.layout import LAParams, LTTextBox
    from pdfminer.pdfinterp import PDFTextExtractionNotAllowed
    import logging 
    logging.Logger.propagate = False 
    logging.getLogger().setLevel(logging.ERROR) 
    praser = PDFParser(open(pdf_path, 'rb'))
    doc = PDFDocument()
    praser.set_document(doc)
    doc.set_parser(praser)
    doc.initialize()
    if not doc.is_extractable:
        raise PDFTextExtractionNotAllowed
    else:
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        content = ''
        for page in doc.get_pages():
            interpreter.process_page(page)                        
            layout = device.get_result()                     
            for x in layout:
                if isinstance(x, LTTextBox):
                    content  = content + x.get_text().strip()
    return content

def pdf_to_audio(pdf_path, rate=125, voice_type_0_or_1=1, read=1, save=0, print_text=0):
    import pyttsx3
    text = pdf_to_text(pdf_path)
    if print_text==1:
        print(text)
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')  
    engine.setProperty('voice', voices[voice_type_0_or_1].id)
    engine.setProperty("rate", rate)
    if save==1:
        import re
        file_name = re.split('[/,\\\]', pdf_path)[-1][:-4]
        engine.save_to_file(text, file_name+'.mp3')
        engine.runAndWait()
        print('MP3 file saved!')
    if read==1:
        engine.say(text)
        engine.runAndWait()