from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import uuid
from src.test_system import DocumentFraudDetectionSystem

app = FastAPI()

# Initialize models once at startup
system = DocumentFraudDetectionSystem(
    yolo_model_path="models/YOLOv8/yolov8_runs/weights/best.pt",
    resnet_model_path="models/ResNet/resnet18_document_classifier_xxx.pth"
)

@app.post("/verify")
async def verify_certificate(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Run through pipeline
        result = system.process_single_image(temp_filename)
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
