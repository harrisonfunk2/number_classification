from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware


from inference import pre_process_image, predict_digit


app = FastAPI()

origins = ['http://localhost:5500', 
           'http://127.0.0.1:5500', 
           'https://harrisonfunk2.github.io',]


app.add_middleware(CORSMiddleware, 
                   allow_origins=origins, 
                   allow_credentials=True, 
                   allow_methods=['*'], 
                   allow_headers=['*'],)

@app.get('/health')
def health():
    return {'status': 'running'}

@app.post('/predict')
async def predict(image: UploadFile = File(...)):
    tensor_img = pre_process_image(image.file)

    if tensor_img is None:
        raise HTTPException(status_code=400, 
                            detail='No digit detected')
    
    output = predict_digit(tensor_img)
    return output
