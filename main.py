import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from database import engine, Base
from routers import home, train, classify, history

app = FastAPI(title="Smile Classifier")

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(home.router)
app.include_router(train.router)
app.include_router(classify.router)
app.include_router(history.router)


@app.on_event("startup")
async def startup():
    Base.metadata.create_all(bind=engine)
    os.makedirs("static/uploaded_images", exist_ok=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
