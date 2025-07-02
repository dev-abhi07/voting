# from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
# from sqlalchemy.orm import Session
# from database import SessionLocal, init_db
# from models import Voter
# import numpy as np
# import faiss
# import uuid
# import os
# import face_recognition
# import io
# from datetime import datetime

# app = FastAPI()

# # ---------- Setup Paths ----------
# IMAGE_DIR = "stored_faces"
# FAISS_INDEX_FILE = "face_index.faiss"
# MAPPING_FILE = "mapping.txt"

# os.makedirs(IMAGE_DIR, exist_ok=True)

# # ---------- Database Init ----------
# init_db()

# # ---------- FAISS Init + Load ----------
# if os.path.exists(FAISS_INDEX_FILE):
#     index = faiss.read_index(FAISS_INDEX_FILE)
#     print("✅ FAISS index loaded from disk")
# else:
#     index = faiss.IndexFlatIP(128)

# if os.path.exists(MAPPING_FILE):
#     with open(MAPPING_FILE, "r") as f:
#         image_mapping = f.read().splitlines()
#     print("✅ Image mapping loaded")
# else:
#     image_mapping = []

# THRESHOLD = 0.92  # Cosine similarity threshold


# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()


# def extract_embedding(image_bytes: bytes) -> np.ndarray:
#     image = face_recognition.load_image_file(io.BytesIO(image_bytes))
#     encodings = face_recognition.face_encodings(image)
#     if not encodings:
#         raise HTTPException(status_code=400, detail="No face detected.")
#     vec = encodings[0].astype("float32")
#     return vec / np.linalg.norm(vec)


# @app.post("/vote")
# async def vote(
#     image: UploadFile = File(...),
#     booth_no: str = Form(...),
#     db: Session = Depends(get_db)
# ):
#     image_bytes = await image.read()

#     try:
#         embedding = extract_embedding(image_bytes).reshape(1, -1)
#     except:
#         raise HTTPException(status_code=400, detail="Invalid image or no face detected.")

#     # Match check
#     if index.ntotal > 0:
#         D, I = index.search(embedding, k=1)
#         if D[0][0] >= THRESHOLD:
#             matched_image = image_mapping[I[0][0]]

#             voter = db.query(Voter).filter(Voter.image_name == matched_image).first()
#             return {
#                 "status": "rejected",
#                 "reason": "Already voted",
#                 "matched_with": matched_image,
#                 "similarity": float(D[0][0]),
#                 "message": f"You have already voted on {voter.created_at.strftime('%Y-%m-%d %I:%M %p')} at booth #{voter.booth_no}"
#             }

#     # New vote: save image
#     filename = f"{uuid.uuid4().hex}.jpg"
#     filepath = os.path.join(IMAGE_DIR, filename)
#     with open(filepath, "wb") as f:
#         f.write(image_bytes)

#     index.add(embedding)
#     image_mapping.append(filename)

#     voter = Voter(
#         image_name=filename,
#         vote_status="success",
#         booth_no=booth_no,
#         created_at=datetime.utcnow()
#     )
#     db.add(voter)
#     db.commit()

#     return {
#         "status": "success",
#         "message": "Vote recorded successfully",
#         "image_saved_as": filename
#     }


# @app.on_event("shutdown")
# def save_faiss_and_mapping():
#     faiss.write_index(index, FAISS_INDEX_FILE)
#     with open(MAPPING_FILE, "w") as f:
#         f.write("\n".join(image_mapping))
#     print("🛑 FAISS index and image mapping saved to disk")


# @app.get("/")
# def root():
#     return {"message": "Face Voting API with persistence is running"}


from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from sqlalchemy.orm import Session
from database import SessionLocal, init_db
from models import Voter
import numpy as np
import faiss
import uuid
import os
import face_recognition
import io
from datetime import datetime
from PIL import Image
import multiprocessing

app = FastAPI()

# ---------- Setup ----------
IMAGE_DIR = "stored_faces"
FAISS_INDEX_FILE = "face_index.faiss"
MAPPING_FILE = "mapping.txt"

os.makedirs(IMAGE_DIR, exist_ok=True)

FAISS_DIM = 128
IVF_CLUSTERS = 100
THRESHOLD = 0.92

# ---------- Database Init ----------
init_db()

# ---------- FAISS Multi-core ----------
faiss.omp_set_num_threads(multiprocessing.cpu_count())

# ---------- FAISS Init ----------
quantizer = faiss.IndexFlatIP(FAISS_DIM)
if os.path.exists(FAISS_INDEX_FILE):
    index = faiss.read_index(FAISS_INDEX_FILE)
    print("✅ FAISS index loaded from disk")
else:
    index = faiss.IndexIVFPQ(quantizer, FAISS_DIM, IVF_CLUSTERS, 8, 8)  # m=8, nbits=8
    print("🆕 New FAISS IVFPQ index created")

# ---------- Mapping Load ----------
if os.path.exists(MAPPING_FILE):
    with open(MAPPING_FILE, "r") as f:
        image_mapping = f.read().splitlines()
    print("✅ Image mapping loaded")
else:
    image_mapping = []

# ---------- DB Dependency ----------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------- Face Embedding ----------
def extract_embedding(image_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        img = img.resize((400, 400))  # Resize improves speed
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        image = face_recognition.load_image_file(io.BytesIO(buf.getvalue()))

    encodings = face_recognition.face_encodings(image)
    if not encodings:
        raise HTTPException(status_code=400, detail="No face detected.")
    vec = encodings[0].astype("float32")
    return vec / np.linalg.norm(vec)

# ---------- Voting API ----------
@app.post("/vote")
async def vote(
    image: UploadFile = File(...),
    booth_no: str = Form(...),
    db: Session = Depends(get_db)
):
    image_bytes = await image.read()

    try:
        embedding = extract_embedding(image_bytes).reshape(1, -1)
    except:
        raise HTTPException(status_code=400, detail="Invalid image or no face detected.")

    # Match existing voter
    if index.is_trained and index.ntotal > 0:
        D, I = index.search(embedding, k=1)
        if D[0][0] >= THRESHOLD:
            matched_image = image_mapping[I[0][0]]
            voter = db.query(Voter).filter(Voter.image_name == matched_image).first()
            if voter:
                return {
                    "status": "rejected",
                    "reason": "Already voted",
                    "matched_with": matched_image,
                    "similarity": float(D[0][0]),
                    "message": f"You have already voted on {voter.created_at.strftime('%Y-%m-%d %I:%M %p')} at booth #{voter.booth_no}"
                }

    # Save new voter
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(IMAGE_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    image_mapping.append(filename)

    voter = Voter(
        image_name=filename,
        vote_status="success",
        booth_no=booth_no,
        created_at=datetime.utcnow()
    )
    db.add(voter)
    db.commit()

    # Train FAISS if needed
    if not index.is_trained and len(image_mapping) >= IVF_CLUSTERS:
        print("⚙️ Training FAISS IVFPQ index...")
        embeddings = []
        for img_file in image_mapping:
            with open(os.path.join(IMAGE_DIR, img_file), "rb") as f:
                emb = extract_embedding(f.read())
                embeddings.append(emb)
        index.train(np.array(embeddings, dtype="float32"))
        index.add(np.array(embeddings, dtype="float32"))
        print("✅ FAISS IVFPQ trained")
    elif index.is_trained:
        index.add(embedding)

    return {
        "status": "success",
        "message": "Vote recorded successfully",
        "image_saved_as": filename
    }

# ---------- Save FAISS & Mapping ----------
@app.on_event("shutdown")
def save_index():
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(MAPPING_FILE, "w") as f:
        f.write("\n".join(image_mapping))
    print("🛑 FAISS index and image mapping saved to disk")

# ---------- Health Check ----------
@app.get("/")
def root():
    return {"message": "Optimized Face Voting API is running"}

