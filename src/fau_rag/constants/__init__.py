from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
API_KEYS = os.getenv('OPENAI_API_KEY')

CONFIG_FILE_PATH = Path("config/config.yaml")

VECTORSTORE_PATH = "knowledgebase/vector_index_fau.faiss"

METADATA_PATH_FAU = "knowledgebase/quality_html-pdf.jsonl"
# FAU chunked data [text, url, file_path, chunk_no, dl_date, chunk_date, quality_score]