import pytesseract
from PIL import Image
import fitz
import io
import os
from bs4 import BeautifulSoup
import pandas as pd
from zipfile import ZipFile
from config import TESSERACT

pytesseract.pytesseract.tesseract_cmd = TESSERACT

class Stream():
    """ Class to store file information as an object. 
        Attributes are data (binary file data), ext (extension), bytesio (BytesIO object), and file_name (file name)
    """
    def __init__(self, **kwargs):
        self.data = None
        self.ext = None
        self.bytesio = None
        self.file_name = None
        self.__dict__.update(kwargs)
        if self.file_name:
            self.file_nameNoExt = os.path.splitext(self.file_name)[0]
            self.ext = os.path.splitext(self.file_name)[-1][1:].upper()

def get_stream(file_path, **kwargs):
    """ Function to get a stream of file-like BytesIO object from a file.
        Function returns Stream object or list of stream objects for every subfile in the file specified (in the case of zip and docx)
    """ 

    if kwargs.get('description'):
        description = kwargs.get('description')
    else:
        description = os.path.splitext(os.path.split(file_path)[-1])[0]
    if kwargs.get('ext'):
        ext = kwargs.get('ext').upper().replace('.','')
    else:
        ext = os.path.splitext(file_path)[-1][1:].upper()
        if ext == '':
            raise TypeError("No file extension found. Must provide value for ext.")
    
    recursive = kwargs.get('recursive')
    if recursive == None:
        recursive = True

    if ext in ['ZIP','DOCX']:
        streams = []
        z = ZipFile(file_path)
        for file in z.namelist():
            s = Stream(file_name=file)
            s.data = z.read(file)
            s.bytesio = io.BytesIO(s.data)
            if s.ext == 'ZIP' and recursive:
                z2 = ZipFile(s.bytesio)
                for file2 in z2.namelist():
                    s1 = Stream(file_name=file2)
                    s1.data = z2.read(file2)
                    s1.bytesio = io.BytesIO(s1.data)
                    streams.append(s1)
                z2.close()
            else:
                streams.append(s)
        z.close()
        if len(streams) == 1:
            return streams[0]
        else:
            return streams
    else:
        s = Stream(file_name=description+'.'+ext.lower(), ext=ext)
        with open(file_path, 'rb') as f:
            s.data = f.read()
        s.bytesio = io.BytesIO(s.data)
        return s

def process_file_arg(obj, **kwargs):
    """ Helper function to manage arguments passed to OCR. 
        Parses stream, file path, and extension from obj and kwargs.
    """
    s = ''
    stream = None
    file_path = None
    ext = ''
    if isinstance(obj, Stream):
        stream = obj
    elif type(obj) == list:
        if len(obj) == 0:
            raise TypeError('Zero-length stream passed')
        elif len(obj) == 1:
            stream = obj[0]
        else:
            stream = obj
    else:
        file_path = obj

    # determine ext
    if kwargs.get('ext'):
        ext = kwargs.get('ext').upper().replace('.','')
    elif file_path:
        ext = os.path.splitext(file_path)[-1][1:].upper()
        if ext == '':
            raise TypeError("No file extension found. Must provide value for ext.")
    elif stream:
        if type(stream) == list:
            for stream_item in stream:
                if stream_item.file_name == 'word/document.xml':
                    ext = 'DOCX'
                    break
            if ext == '':
                raise TypeError('Unable to tell file extension. Pass value to ext.')
        elif stream.ext:
            ext = stream.ext
        else:
            raise TypeError("No file extension found. Must provide value for ext.")

    return (stream, file_path, ext)
    
def get_text(obj=None, **kwargs): #
    """ Function to extract text from file
        Function can accept a file path or a Stream object. ext must be specified if using a BytesIO object 
    """
    s = ''
    dpi = 300
    if kwargs.get('dpi'):
        dpi = kwargs.get('dpi')
    
    force_ocr = False
    if kwargs.get('force_ocr'):
        force_ocr = kwargs.get('force_ocr')
    stream, file_path, ext = process_file_arg(obj, **kwargs)

    if ext == 'PDF':
        if file_path:
            doc = fitz.open(file_path)
        elif stream:
            doc = fitz.open(stream=stream.bytesio, filetype="pdf")
        page2len = 0
        if doc.page_count > 1:
            page2len = len(doc.load_page(1).get_text())
        if len(doc.load_page(0).get_text()) + page2len > 952 and force_ocr == False:
            for page in doc:
                s += page.get_text()
        else:
            return ''
        
    elif ext in ['XLS','XLSX']:
        if file_path:
            s = pd.read_excel(file_path).to_csv()
        elif stream:
            s = pd.read_excel(stream.bytesio).to_csv()
    
    elif ext in ['CSV']:
        if file_path:
            s = pd.read_csv(file_path).to_csv()
        elif stream:
            s = pd.read_csv(stream.bytesio).to_csv()
        
    elif ext == 'DOCX':
        data = None
        if file_path:
            for st in get_stream(file_path,ext='DOCX'):
                if st.file_name == 'word/document.xml':
                    data = st.data
        elif stream:
            if type(stream) == list:
                for stream_item in stream:
                    if stream_item.file_name == 'word/document.xml':
                        data = stream_item.data
            else:
                if stream.file_name == 'word/document.xml':
                    data = stream.data
                else:
                    raise ValueError('Must pass list of streams or word/document.xml if using stream for DOCX.')
        if data:
            soup = BeautifulSoup(data, "lxml")
            for script in soup(["script", "style"]):
                script.extract()
            s += soup.get_text()

        
    elif ext in ['HTM','HTML']:
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as f:
                html = f.read()
        elif stream:
            html = stream.bytesio.getvalue()
        
        soup = BeautifulSoup(html, features="html.parser")
        for script in soup(["script", "style"]):
            script.extract()
        s += soup.get_text()
    
    elif ext in ['TIF','JPG','PNG','JPEG','TIFF']:
        return ''
    
    elif ext == 'TXT':
        if file_path:
            with open(file_path, 'r') as f:
                s = f.read()
        elif stream:
            s = stream.bytesio.getvalue().decode()
        
    else:
        return ''
    
    return s

def OCR(obj=None, **kwargs):
    """ Function to perform OCR on a file file
        Function can accept a file path or a Stream object. ext must be specified if using a BytesIO object 
    """
    s = ''
    dpi = 300
    if kwargs.get('dpi'):
        dpi = kwargs.get('dpi')
    
    stream, file_path, ext = process_file_arg(obj, **kwargs)

    if ext == 'PDF':
        if file_path:
            doc = fitz.open(file_path)
        elif stream:
            doc = fitz.open(stream=stream.bytesio, filetype="pdf")
        
        zoom = dpi / 72
        magnify = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=magnify)
            data=pix.tobytes("png")
            img = Image.open(io.BytesIO(data))
            s += pytesseract.image_to_string(img)
    
    elif ext in ['TIF','JPG','PNG','JPEG','TIFF']:
        if file_path:
            img = Image.open(file_path)
        elif stream:
            img = Image.open(stream.bytesio)
        s += pytesseract.image_to_string(img)
        
    return s
    
