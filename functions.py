import re
import os
import openai
import subprocess
import json
import sqlite3
import markdown
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
import requests
import pytesseract
from PIL import Image
import numpy as np

# Load environment variables
load_dotenv()
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
AIPROXY_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
EMAIL = os.getenv("EMAIL")
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)



# A1: Run datagen.py to generate required data
def run_datagen():
    """Downloads and runs datagen.py, storing results in ./data."""
    datagen_url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
    subprocess.run(["uv", "run", datagen_url, EMAIL, "--root", str(DATA_DIR)], check=True)

# A2: Format markdown using Prettier
def format_markdown():
    """Formats /data/format.md using Prettier."""
    subprocess.run(["npx", "prettier", "--write", str(DATA_DIR / "format.md")], check=True) #node problem create karra hai


# A3:  Count the number of Wednesdays
DATE_FORMATS = [
    "%Y-%m-%d",
    "%d-%b-%Y",
    "%d/%m/%Y",
    "%b %d, %Y",       
    "%Y/%m/%d",        
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%d-%b-%Y %H:%M:%S"
]

def parse_date(date_str):
    """Attempts to parse a date string using multiple formats."""
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {date_str}")

def split_concatenated_dates(line):
    """Splits concatenated dates using regex (e.g., 'Dec 15, 2012Dec 15, 2016')."""
    return re.findall(r"[A-Za-z]{3} \d{1,2}, \d{4}|\d{4}[-/]\d{2}[-/]\d{2}", line)

def count_wednesdays():
    """Counts Wednesdays in /data/dates.txt and writes the count to /data/dates-wednesdays.txt."""
    input_file = DATA_DIR / "dates.txt"
    output_file = DATA_DIR / "dates-wednesdays.txt"

    try:
        with open(input_file, "r") as f:
            lines = f.readlines()

        wednesday_count = 0

        for line in lines:
            # Handle concatenated dates
            dates = split_concatenated_dates(line)
            for date in dates:
                if parse_date(date).weekday() == 2: 
                    wednesday_count += 1

        # Write result
        with open(output_file, "w") as f:
            f.write(str(wednesday_count) + "\n")

    except Exception as e:
        print(f"❌ Error: {e}")

# A4: Sort contacts.json by last_name and first_name
def sort_contacts():
    """Sorts contacts by last_name, then first_name, and writes to contacts-sorted.json."""
    input_file = DATA_DIR / "contacts.json"
    output_file = DATA_DIR / "contacts-sorted.json"

    try:
        # Load JSON file
        with open(input_file, "r", encoding="utf-8") as f:
            contacts = json.load(f)

        # Validate data format
        if not isinstance(contacts, list) or not all(isinstance(c, dict) for c in contacts):
            raise ValueError("Invalid JSON format: Expected a list of dictionaries")

        # Sort by last_name, then first_name
        sorted_contacts = sorted(contacts, key=lambda c: (c.get("last_name", ""), c.get("first_name", "")))

        # Write to output file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(sorted_contacts, f, indent=2)

        print(f"✅ Contacts sorted successfully! Output saved to {output_file}")

    except Exception as e:
        print(f"❌ Error: {e}")

# A5: Write the first line of the 10 most recent .log file
def extract_recent_logs():
    LOGS_DIR = DATA_DIR / "logs"
    """Writes the first line of the 10 most recent .log files to logs-recent.txt."""
    output_file = DATA_DIR / "logs-recent.txt"

    try:
        # Get all .log files sorted by modification time (most recent first)
        log_files = sorted(LOGS_DIR.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)

        # Take the 10 most recent logs
        recent_logs = log_files[:10]

        results = []
        for log_file in recent_logs:
            try:
                # Read the first line
                with open(log_file, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    results.append(first_line)
            except Exception as e:
                print(f"❌ Error reading {log_file}: {e}")

        # Write to logs-recent.txt
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(results) + "\n")

        print(f"✅ First lines of 10 most recent logs saved to {output_file}")

    except Exception as e:
        print(f"❌ Error: {e}")

# A6: Extract markdown headings
def index_markdown_titles():
    DOCS_DIR = DATA_DIR / "docs"
    """Creates an index.json mapping each Markdown file to its first H1 title."""
    output_file = DOCS_DIR / "index.json"
    index_data = {}

    try:
        # Find all .md files in /data/docs/
        md_files = list(DOCS_DIR.rglob("*.md"))

        for md_file in md_files:
            try:
                # Read the file line by line
                with open(md_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("# "):  # First-level heading
                            title = line[2:].strip()  # Extract title text
                            relative_path = md_file.relative_to(DOCS_DIR).as_posix()
                            index_data[relative_path] = title
                            break  # Only take the first H1 and stop reading

            except Exception as e:
                print(f"❌ Error reading {md_file}: {e}")

        # Save the index as JSON
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=4)

        print(f"✅ Markdown index saved to {output_file}")

    except Exception as e:
        print(f"❌ Error: {e}")

# A7: Extract Email Sender Using an LLM
def extract_email_sender():
    """Extracts the sender's email address from email.txt using AI Proxy."""
    email_file = DATA_DIR / "email.txt"
    output_file = DATA_DIR / "email-sender.txt"

    # Read email content
    try:
        with open(email_file, "r", encoding="utf-8") as f:
            email_content = f.read().strip()
    except FileNotFoundError:
        print(f"❌ Error: {email_file} not found.")
        return
    except Exception as e:
        print(f"❌ Error reading {email_file}: {e}")
        return

    if not email_content:
        print("❌ Error: Email file is empty.")
        return

    # Call AI Proxy for extraction
    try:
        headers = {
            "Authorization": f"Bearer {AIPROXY_TOKEN}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "Extract only the sender's email address from the following email."},
                {"role": "user", "content": email_content}
            ],
            "temperature": 0
        }
        
        response = requests.post(AIPROXY_URL, json=data, headers=headers)
        response.raise_for_status()
        sender_email = response.json()["choices"][0]["message"]["content"].strip()

        # Validate extracted email
        if "@" not in sender_email or "." not in sender_email:
            print("❌ Error: Extracted email is not valid.")
            return

        # Write extracted email to output file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(sender_email + "\n")

        print(f"✅ Extracted sender email saved to {output_file}")

    except requests.exceptions.RequestException as e:
        print(f"❌ AI Proxy Request Error: {e}")
    except KeyError:
        print(f"❌ Unexpected response format: {response.text}")

# A8: Extract credit card number
def extract_text_from_image(image_path):
    """Extracts raw text from an image using Tesseract OCR."""
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"❌ Error extracting text from image: {e}")
        return None

def refine_credit_card_number(ocr_text):
    """Passes OCR-extracted text to AI Proxy to extract a valid credit card number."""
    try:
        headers = {
            "Authorization": f"Bearer {AIPROXY_TOKEN}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "Extract only the credit card number from the following OCR output and return it without spaces."},
                {"role": "user", "content": ocr_text}
            ],
            "temperature": 0
        }
        
        response = requests.post(AIPROXY_URL, json=data, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    except requests.exceptions.RequestException as e:
        print(f"❌ AI Proxy Request Error: {e}")
        return None
    except KeyError:
        print(f"❌ Unexpected response format: {response.text}")
        return None

def process_credit_card_image():
    """Extracts a credit card number from an image and saves it to a file."""
    IMAGE_FILE = DATA_DIR / "credit_card.png"
    OUTPUT_FILE = DATA_DIR / "credit-card.txt"

    if not IMAGE_FILE.exists():
        print(f"❌ Error: {IMAGE_FILE} not found.")
        return
    
    print(f"🔍 Processing {IMAGE_FILE}...")

    # Step 1: Extract text using OCR
    ocr_text = extract_text_from_image(IMAGE_FILE)
    if not ocr_text:
        return

    print(f"📝 OCR Output: {ocr_text}")

    # Step 2: Use AI Proxy to extract the credit card number
    card_number = refine_credit_card_number(ocr_text)
    if not card_number:
        return

    # Step 3: Save the extracted number to a file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(card_number + "\n")

    print(f"✅ Extracted credit card number saved to {OUTPUT_FILE}")

# A9: Similar comments
def get_embedding(text):
    AIPROXY_URL='https://aiproxy.sanand.workers.dev/openai/v1/embeddings'
    """Fetches the embedding of a given text from AI Proxy."""
    try:
        headers = {
            "Authorization": f"Bearer {AIPROXY_TOKEN}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "text-embedding-3-small",
            "input": text
        }
        
        response = requests.post(AIPROXY_URL, json=data, headers=headers)
        response.raise_for_status()
        
        return response.json()["data"][0]["embedding"]
    
    except requests.exceptions.RequestException as e:
        print(f"❌ AI Proxy Request Error: {e}")
        return None
    except KeyError:
        print(f"❌ Unexpected response format: {response.text}")
        return None

def cosine_similarity(vec1, vec2):
    """Computes cosine similarity between two vectors."""
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_most_similar_comments():
    INPUT_FILE = DATA_DIR / "comments.txt"
    OUTPUT_FILE = DATA_DIR / "comments-similar.txt"
    
    if not INPUT_FILE.exists():
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    # Read comments
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        comments = [line.strip() for line in f.readlines() if line.strip()]
    
    if len(comments) < 2:
        print("❌ Not enough comments to find a similar pair.")
        return
    
    print(f"📄 Loaded {len(comments)} comments.")

    # Compute embeddings
    embeddings = []
    for comment in comments:
        emb = get_embedding(comment)
        if emb:
            embeddings.append((comment, emb))
        else:
            print(f"❌ Skipping comment due to missing embedding: {comment}")
    
    if len(embeddings) < 2:
        print("❌ Not enough valid embeddings to compare.")
        return

    # Find most similar pair
    most_similar_pair = None
    max_similarity = -1

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = cosine_similarity(embeddings[i][1], embeddings[j][1])
            if sim > max_similarity:
                max_similarity = sim
                most_similar_pair = (embeddings[i][0], embeddings[j][0])

    if not most_similar_pair:
        print("❌ No similar comments found.")
        return

    # Write result to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(most_similar_pair) + "\n")

    print(f"✅ Most similar comments saved to {OUTPUT_FILE}")

# A10: Gold database
def calculate_gold_ticket_sales():
    db_path = DATA_DIR / "ticket-sales.db"
    output_file = DATA_DIR / "ticket-sales-gold.txt"

    if not db_path.exists():
        print(f"❌ Error: Database file {db_path} not found.")
        return
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query to calculate total sales for 'Gold' ticket type
        cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
        total_sales = cursor.fetchone()[0]

        # If no Gold tickets exist, set total sales to 0
        if total_sales is None:
            total_sales = 0

        # Write the total sales to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(str(total_sales) + "\n")

        print(f"✅ Total Gold ticket sales saved to {output_file}")

    except sqlite3.Error as e:
        print(f"❌ Database Error: {e}")

    finally:
        conn.close()











# run_datagen()         WORKING!
# format_markdown()
# count_wednesdays()    WORKING!
# sort_contacts()       WORKING!
# extract_recent_logs() WORKING!
# index_markdown_titles() WORKING!
# extract_email_sender() WORKING!
# process_credit_card_image() WORKING!
# find_most_similar_comments() PARTIALLY WORKING!
# calculate_gold_ticket_sales() WORKING!

