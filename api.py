from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.encoders import jsonable_encoder
from database import Database, User, UserIn, Block, Project
import uvicorn

from fastapi import FastAPI, HTTPException


app = FastAPI(debug=True)

@app.get("/")
def root() -> FileResponse:
    return FileResponse("index.html")

@app.get("/registration")
def regisration() -> FileResponse:
    return FileResponse("registration.html")


# Обработчик для создания пользователя
@app.post("/registration")
async def create_user(user: User) -> JSONResponse:
    database = Database('server')
    try:
        user = database.add_user(user)
        return JSONResponse(jsonable_encoder(user))
    except Exception as e:
        raise HTTPException(status_code=401, detail=e)


# Обработчик для получения всех пользователей
@app.get("/users")
async def get_all_users() -> JSONResponse:
    database = Database('server')
    users = database.get_all_users()
    return JSONResponse(jsonable_encoder(users))

# Обработчик для проверки существования аккаунта
@app.post("/login")
async def login(user: UserIn) -> JSONResponse:
    database = Database('server')
    existing_user = database.get_user(user)
    if existing_user:
        user_id = existing_user.id
        user_blocks = database.get_user_blocks(user_id)
        user_projects = database.get_user_projects(user_id)
        return JSONResponse(jsonable_encoder({'user': existing_user, 'blocks': user_blocks, 'projects': user_projects}))
    else:
        raise HTTPException(status_code=401, detail="Incorrect username or password.")


# Обработчик для загрузки профиля
# @app.post("/profile")
# async def login(user_id: int) -> JSONResponse:
#     database = Database('server')
#     user_blocks = database.get_user_blocks(user_id)
#     user_projects = database.get_user_projects(user_id)
#     return JSONResponse(jsonable_encoder({'blocks': user_blocks, 'projects': user_projects}))


# Обработчик для добавления блока
@app.post("/add_block")
async def login(block: Block) -> JSONResponse:
    database = Database('server')
    try:
        blocks = database.add_user_block(block)
        return JSONResponse(jsonable_encoder(blocks))
    except Exception as e:
        raise HTTPException(status_code=401, detail=e)

# Обработчик для добавления проекта
@app.post("/add_project")
async def login(project: Project) -> JSONResponse:
    database = Database('server')
    try:
        projects = database.add_user_block(project)
        return JSONResponse(jsonable_encoder(projects))
    except Exception as e:
        raise HTTPException(status_code=401, detail=e)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)