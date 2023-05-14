import uvicorn

from typing import *
from jose import JWTError, jwt
from fastapi import FastAPI, Body
from collections import defaultdict
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from database import Database, User, UserIn, Block, Project, UserOut


app = FastAPI(debug=True)

SECRET_KEY = "mysecretkey"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
origins = [
    "http://127.0.0.1:5173/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = """
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    
    def __init__(self) -> None:
        super(Model, self).__init__()
        
        self.layers = nn.Sequential(
{}
        )
    
    
    def forward(self, data):
        return self.layers(data)
"""

def create_access_token(data: dict, expires_delta):
    print("create token")
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str) -> UserOut:
    print('get_current_user')
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("username")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    database = Database('server')
    user = database.get_user(username)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return user


# Обработчик для создания пользователя
@app.post("/api/registration")
async def create_user(user: User) -> JSONResponse:
    database = Database('server')
    try:
        user = database.add_user(user)
        access_token = create_access_token(
            data={"username": user.username}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        return JSONResponse(jsonable_encoder({'token': access_token, 'user': user}))
    except Exception as e:
        raise HTTPException(status_code=401, detail=e)


# Обработчик для проверки существования аккаунта
@app.post("/api/login")
async def login(user: UserIn) -> JSONResponse:
    print('login')
    database = Database('server')
    existing_user = database.auth_user(user)
    if existing_user:
        access_token = create_access_token(
            data={"username": existing_user.username}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        user_id = existing_user.id
        user_blocks = database.get_user_blocks(user_id)
        user_projects = database.get_user_projects(user_id)
        return JSONResponse(jsonable_encoder({'token': access_token,'user': existing_user, 'blocks': user_blocks, 'projects': user_projects}))
    else:
        raise HTTPException(status_code=401, detail="Incorrect username or password.")


# Обработчик для загрузки профиля
@app.post("/token")
async def profile(current_user: Annotated[UserOut, Depends(get_current_user)]) -> JSONResponse:
    print('profile')
    database = Database('server')
    user_blocks = database.get_user_blocks(current_user.id)
    user_projects = database.get_user_projects(current_user.id)
    return JSONResponse(jsonable_encoder({'user': current_user, 'blocks': user_blocks, 'projects': user_projects}))



def modify_objects(nodes, edges):
    new_nodes = defaultdict(dict)
    for node in nodes:
        new_nodes.update({
                int(node['id']): {
                    'name': node['data'].pop('label'), 
                    'params': node['data']
                }
            }
        )
    
    new_edges = defaultdict(dict)
    for edge in edges:
        new_edges.update({
                int(edge['source']): {
                    'id': edge['id'], 
                    'target': int(edge['target'])
                }
            }
        )
    return new_nodes, new_edges


# Обработчик для создания кода из блоков
@app.post("/api/create_code")
async def create_code(data: dict = Body(...)):
    nodes = data['instance']['nodes']
    edges = data['instance']['edges']
    nodes, edges = modify_objects(nodes, edges)
    curr_node = 0
    s = ''
    while len(edges) > 0:
        curr_node = edges.pop(curr_node)['target']
        s += f"\t\t\t{nodes[curr_node]['name']}({', '.join(f'{key}={value}' for key, value in nodes[curr_node]['params'].items())})"
        if len(edges) > 0:
            s += '\n'
        # curr_node = edges.pop(curr_node)['target']
    
    with open('model.py', 'w') as f:
        f.write(model.format(s))
    return FileResponse('model.py')


# Обработчик для добавления блока
@app.post("/api/add_block")
async def add_block(block: Block) -> JSONResponse:
    database = Database('server')
    try:
        blocks = database.add_user_block(block)
        return JSONResponse(jsonable_encoder(blocks))
    except Exception as e:
        raise HTTPException(status_code=401, detail=e)


# Обработчик для добавления проекта
@app.post("/api/add_project")
async def add_project(project: Project) -> JSONResponse:
    database = Database('server')
    try:
        projects = database.add_user_project(project)
        return JSONResponse(jsonable_encoder(projects))
    except Exception as e:
        raise HTTPException(status_code=401, detail=e)



if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)