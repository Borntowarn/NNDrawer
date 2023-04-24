from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.encoders import jsonable_encoder
from database import Database, User, UserIn
import uvicorn

from fastapi import FastAPI, HTTPException


app = FastAPI(debug=True)

@app.get("/")
def root():
    return FileResponse("index.html")


# Обработчик для создания пользователя
@app.get("/registration")
async def create_user():
    return FileResponse("registration.html")


# Обработчик для создания пользователя
@app.post("/registration")
async def create_user(user: User):
    database = Database('server')
    try:
        user = database.add_user(user)
        return JSONResponse(jsonable_encoder(user))
    except Exception as e:
        raise HTTPException(status_code=401, detail=e)


# Обработчик для получения всех пользователей
@app.get("/users")
async def get_all_users():
    database = Database('server')
    users = database.get_all_users()
    return JSONResponse(jsonable_encoder(users))

# Обработчик для проверки существования аккаунта
@app.post("/login")
async def login(user: UserIn):
    database = Database('server')
    existing_user = database.get_user(user)
    if existing_user:
        return JSONResponse(jsonable_encoder(existing_user))
    else:
        raise HTTPException(status_code=401, detail="Incorrect username or password.")


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)