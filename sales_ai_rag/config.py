from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env in development
load_dotenv(dotenv_path=Path(__file__).with_suffix('.env'), override=False)

# Google Generative AI
GEMINI_MODEL: str = "gemini-2.5-flash"
GOOGLE_API_KEY: str = "AIzaSyAUoA-BTVct0t3GtkjW0uKCN4G_iczgBTU"  # required
EMBED_MODEL: str = "models/embedding-001"

# Qdrant
QDRANT_URL: str = "https://8243ef5f-44fa-46cf-ac01-057b8e587574.europe-west3-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.TgnmoSw0DIaIX_hwflTx2cWSPMhPcLrLFJ6tD_RwtDk"
QDRANT_COLLECTION: str = "theory"

# Chunking
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 100)) # Default chunk size
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 25))    # Default chunk overlap   