from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
import google.generativeai as genai
import requests
from flask_cors import CORS  
import fitz
from docx import Document
import json
from PIL import Image
import pytesseract
import gdown
from io import BytesIO
import io
import re

load_dotenv()

app = Flask(__name__)
CORS(app) 

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Gemini API key is not set in environment variables.")

genai.configure(api_key=GEMINI_API_KEY)


KEY=os.getenv("Key_value")
print(KEY)

API_KEY = os.getenv("OCR_SPACE_API_KEY")
API_URL = "https://api.ocr.space/parse/image"


languages = {
    "Arabic": "ara",
    "Bulgarian": "bul",
    "Chinese (Simplified)": "chs",
    "Chinese (Traditional)": "cht",
    "Croatian": "hrv",
    "Czech": "cze",
    "Danish": "dan",
    "Dutch": "dut",
    "English": "eng",
    "Finnish": "fin",
    "French": "fre",
    "German": "ger",
    "Greek": "gre",
    "Hungarian": "hun",
    "Korean": "kor",
    "Italian": "ita",
    "Japanese": "jpn",
    "Polish": "pol",
    "Portuguese": "por",
    "Russian": "rus",
    "Slovenian": "slv",
    "Spanish": "spa",
    "Swedish": "swe",
    "Thai": "tha",
    "Turkish": "tur",
    "Ukrainian": "ukr",
    "Vietnamese": "vnm"
}

def get_language_code(language_name):
    return languages.get(language_name, "Language not found")


def download_file(url):
    """
    Downloads a file from a given URL and saves it locally.
    Supports Google Drive and Dropbox links.
    Returns the local file path.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        if "drive.google.com" in url:
            file_id = url.split("/d/")[1].split("/")[0]  
            download_url = f"https://drive.google.com/uc?id={file_id}"
            save_path = "./downloaded_file.pdf"
            gdown.download(download_url, save_path, quiet=False)
            return save_path

        if "dropbox.com" in url:
            url = url.replace("?dl=0", "?dl=1") 

        response = requests.get(url, stream=True, headers=headers, allow_redirects=True)

        if response.status_code == 200:
            ext = url.split('.')[-1].split("?")[0] 
            save_path = f"./downloaded_file.{ext}"

            with open(save_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)

            return save_path
        else:
            print(f"HTTP Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Download failed: {e}")
        return None

def ocr_space_file(image_bytes, language="eng", filetype="JPG"):
    """
    Sends an image file (in bytes) to the OCR API and returns the extracted text.
    """
    payload = {
        "apikey": API_KEY,
        "language": language,
        "isOverlayRequired": False,
        "filetype": filetype
    }
    try:
        response = requests.post(
            API_URL,
            files={"file": ("image.jpg", image_bytes, "image/jpeg")},
            data=payload,
        )
        result = response.json()

        print("OCR API Response:", result)  

        if result.get("IsErroredOnProcessing", False):
            return {"error": result.get("ErrorMessage", "Unknown error")}

        if "ParsedResults" not in result or not result["ParsedResults"]:
            return {"error": "No text extracted or invalid response format"}

        return {"text": result["ParsedResults"][0].get("ParsedText", "")}

    except Exception as e:
        return {"error": str(e)}


def extract_text(file_path,language,use_openocr=True):
    """
    Extracts text from a given file:
    - PDF: Converts pages to images using PyMuPDF, then uses OpenOCR or Tesseract.
    - DOCX: Uses `python-docx` to extract text.
    - Images: Uses OpenOCR or Tesseract OCR.
    - TXT: Reads the plain text.
    """
    text = ""
    try:
        if file_path.endswith(".pdf"):
            doc = fitz.open(file_path)
            for page_num, page in enumerate(doc):
                pix = page.get_pixmap()
                img_byte_arr = io.BytesIO()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img.save(img_byte_arr, format="JPEG", quality=80)
                img_bytes = img_byte_arr.getvalue()

                extracted_result = (
                    ocr_space_file(img_bytes, language, filetype="JPG") if use_openocr
                    else {"text": pytesseract.image_to_string(img)}
                )

                if "error" in extracted_result:
                    return extracted_result 

                extracted_text = extracted_result["text"]
                if extracted_text:
                    text += f"Page {page_num + 1}:\n{extracted_text}\n\n"

        elif file_path.endswith(".docx"):
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])

        elif file_path.endswith((".png", ".jpg", ".jpeg")):
            with open(file_path, "rb") as img_file:
                img_bytes = img_file.read()
                extracted_result = (
                    ocr_space_file(img_bytes,language,filetype="JPG") if use_openocr
                    else {"text": pytesseract.image_to_string(Image.open(file_path))}
                )

                if "error" in extracted_result:
                    return extracted_result  

                text = extracted_result["text"]

        else:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

    except Exception as e:
        return {"error": str(e)}

    return text.strip()

def extract_text_from_url(url,language="eng"):
    """
    Downloads the file from the URL and extracts text.
    """
    file_path = download_file(url)
    if file_path:
        print("downloaded")
        return extract_text(file_path,language)
    return ""

import json
import re

def fix_json(json_str):
    """
    Fix common JSON syntax errors in the provided text
    """
    # Step 1: Clean hidden invisible characters
    json_str = json_str.encode('utf-8', 'ignore').decode('utf-8')  # remove weird unicode
    json_str = json_str.replace('\r\n', '\n').replace('\r', '\n')   # normalize newlines
    json_str = re.sub(r'[^\x20-\x7E\n\t]', '', json_str)            # remove non-printable except newline and tab

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON Error: {e}")

        # Fallback repairing steps...
        position = e.pos
        if "Expecting ',' delimiter" in str(e):
            fixed_text = json_str[:position] + ',' + json_str[position:]
            try:
                return json.loads(fixed_text)
            except json.JSONDecodeError as e2:
                print(f"Still having errors after fixing: {e2}")

        try:
            fixed_text = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', json_str)
            return json.loads(fixed_text)
        except:
            pass

        try:
            fixed_text = re.sub(r',\s*}', '}', json_str)
            fixed_text = re.sub(r',\s*]', ']', fixed_text)
            return json.loads(fixed_text)
        except:
            pass

        try:
            import demjson3
            return demjson3.decode(json_str)
        except ImportError:
            print("For more robust JSON fixing, install demjson3: pip install demjson3")
        except Exception as e3:
            print(f"Failed to fix JSON: {e3}")

    return None


@app.route('/generate_initial_report', methods=['POST'])
def generate_initial_report():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON input"}), 400
        
        metadata = {item_key: item_value for entry in data.get("metadata", []) for item_key, item_value in entry.items()}
        language = metadata.get("language", "English")
        specific_constraints = metadata.get("Specific_constraint", "")
        Key=metadata.get("Key", "")
        category=metadata.get("category", "")

        lang=get_language_code(language)

        if(KEY!=Key):
            return jsonify({"error": "Unauthorized access"}), 400 
        
        if not language:
            return jsonify({"error": "language is required"}), 400

        extracted_text = ""
        for file_entry in data.get("file_data", []):
            if file_entry.get("t") == "files" and file_entry.get("a"):
                extracted_content = extract_text_from_url(file_entry["a"],lang)
                extracted_text += f"{file_entry['q']} {extracted_content}\n"  
            else:
                extracted_text += f"{file_entry.get('q', '')} {file_entry.get('a', '')}\n"
        
        if not extracted_text.strip():
           return jsonify({"error": "No input text found"}), 400

        
        Text = ""
        with open("initial_report_final.txt", "r", encoding="utf-8") as file:
            Text = file.read().replace('\uf0a7', '-').strip()

        prompt = (
        f"Follow These Instructions strictly:\n\n"
        f"{Text}\n\n"
        f"Generate A Report based on the Following Context:\n\n"
        f"CONTEXT :{extracted_text}\n\n"
        f"Specific Instruction:{specific_constraints}\n\n"
        f"****Language of Response Must be{language}****\n\n"
        f"Surf the internet to Give reference links.\n\n"
    ) 
        
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        
        print(response.text)
        json_str = response.text.strip().replace('```json', '').replace('```', '').strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            print("Standard JSON parsing failed, attempting to fix JSON errors...")
            data = fix_json(json_str)
        
        if data is None:
            print("Could not parse or fix JSON")
            return {"error": "Invalid JSON format"}, 400

        print(data)
        return data, 200
    
    except Exception as e:
        print("❌ Error processing request:", str(e))
        return jsonify({"error": "Internal Server Error"}), 500
    

@app.route('/generate_preliminary_report', methods=['POST'])
def generate_preliminary_report():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON input"}), 400
        
        metadata = {item_key: item_value for entry in data.get("metadata", []) for item_key, item_value in entry.items()} 
        language = metadata.get("language", "English")
        specific_constraints = metadata.get("Specific_constraint", "")
        Key=metadata.get("Key", "")
        category=metadata.get("category", "")

        lang=get_language_code(language)

        if(KEY!=Key):
            return jsonify({"error": "Unauthorized access"}), 400 
        
        if not language:
            return jsonify({"error": "Category and language are required"}), 400

        extracted_text = ""

        extracted_text = ""
        for file_entry in data.get("file_data", []):
            if file_entry.get("t") == "files" and file_entry.get("a"):
                extracted_content = extract_text_from_url(file_entry["a"],lang)
                extracted_text += f"{file_entry['q']} {extracted_content}\n"  
            else:
                extracted_text += f"{file_entry.get('q', '')} {file_entry.get('a', '')}\n"
        print(extracted_text)

        Text=""
        with open("preliminaryReport.txt", "r", encoding="utf-8") as file:
            Text = [line.strip().replace('\uf0a7', '-') for line in file]
        

        prompt = (
         f"**Follow the instructions given below:\n\n**"
         f"{Text}\n\n"
         f"**Generate the Report using the Context Given Below**\n\n"
         f"CONTEXT : {extracted_text}\n\n"
         f"Specific Instructions: {specific_constraints}\n\n"
         f"****Language of Response Must be{language}****\n\n"
         f"Surf the internet to Give reference links."
    )
        print(prompt)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)

        print(response)
       
        return jsonify({"response": response.text}), 200
    
    except Exception as e:
        print("❌ Error processing request:", str(e))
        return jsonify({"error": "Internal Server Error"}), 500
    

@app.route('/generate_final_report', methods=['POST'])
def generate_final_report():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON input"}), 400
        
        metadata = {item_key: item_value for entry in data.get("metadata", []) for item_key, item_value in entry.items()} 
        language = metadata.get("language", "English")
        specific_constraints = metadata.get("Specific_constraint", "")
        Key=metadata.get("Key", "")
        category=metadata.get("category", "")

        lang=get_language_code(language)

        if(KEY!=Key):
            return jsonify({"error": "Unauthorized access"}), 400 
        
        if not language or not category:
            return jsonify({"error": "Category and language are required"}), 400

        extracted_text = ""

        extracted_text = ""
        for file_entry in data.get("file_data", []):
            if file_entry.get("t") == "files" and file_entry.get("a"):
                extracted_content = extract_text_from_url(file_entry["a"],lang)
                extracted_text += f"{file_entry['q']} {extracted_content}\n"  
            else:
                extracted_text += f"{file_entry.get('q', '')} {file_entry.get('a', '')}\n"
        print(extracted_text)

        if not extracted_text.strip():
           return jsonify({"error": "No Input text found"}), 400

        
        Text = ""
        with open("finalReport_next.txt", "r", encoding="utf-8") as file:
            Text = file.read().replace('\uf0a7', '-').strip()
        

        prompt = (
         f"You are a powerful and intelligent Report Generator\n\n"
         f"CONTEXT for Report:{extracted_text}\n\n"
         f"**Follow the instructions given below:\n\n**"
         f"**Generate only the Content strictly in Given format with very detailed information.**\n\n"
         f"{Text}\n\n"
         f"Specific Instructions: {specific_constraints}\n\n"
         f"****Language of Response Must be{language}****\n\n"
         f"Surf the internet to Give reference links.\n\n"
    )
        print(prompt)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)

        print(response.text)
        json_str = response.text.strip().replace('```json', '').replace('```', '').strip()

        data = json.loads(json_str)

        print(data)

        return data,200
        
    except Exception as e:
        print("❌ Error processing request:", str(e))
        return jsonify({"error": "Internal Server Error"}), 500
    

@app.route('/generate_Audit_report', methods=['POST'])
def generate_Audit_report():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON input"}), 400
        
        metadata = {item_key: item_value for entry in data.get("metadata", []) for item_key, item_value in entry.items()} 
        language = metadata.get("language", "English")
        specific_constraints = metadata.get("Specific_constraint", "")
        Key=metadata.get("Key", "")
        category=metadata.get("category", "")

        lang=get_language_code(language)

        if(KEY!=Key):
            return jsonify({"error": "Unauthorized access"}), 400 
        
        if not language:
            return jsonify({"error": "Category and language are required"}), 400

        extracted_text = ""

        extracted_text = ""
        for file_entry in data.get("file_data", []):
            if file_entry.get("t") == "files" and file_entry.get("a"):
                extracted_content = extract_text_from_url(file_entry["a"],lang)
                extracted_text += f"{file_entry['q']} {extracted_content}\n"  
            else:
                extracted_text += f"{file_entry.get('q', '')} {file_entry.get('a', '')}\n"
        print(extracted_text)

        Text = ""
        with open("AuditChecklist.txt", "r", encoding="utf-8") as file:
            Text = file.read().replace('\uf0a7', '-').strip()

        

        prompt = (
         f"**Follow the instructions given below:\n\n**"
         f"{Text}\n\n"
         f"CONTEXT for Audit Checklist:{extracted_text}\n\n"
         f"Specific Instructions: {specific_constraints}\n\n"
         f"****Language of Response Must be{language}****\n\n"
    )
        print(prompt)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)

        print(response.text)
        json_str = response.text.strip().replace('```json', '').replace('```', '').strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            print("Standard JSON parsing failed, attempting to fix JSON errors...")
            data = fix_json(json_str)
        
        if data is None:
            print("Could not parse or fix JSON")
            return {"error": "Invalid JSON format"}, 400

        print(data)
        return data, 200
    
    except Exception as e:
        print("❌ Error processing request:", str(e))
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(debug=True)
