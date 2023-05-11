from fastapi import FastAPI, Form, Body, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from database import Database, User, UserIn, Block, Project
import json
import uvicorn
from pydantic import Json
from fastapi import FastAPI, HTTPException
from collections import defaultdict


app = FastAPI(debug=True)

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

@app.get("/api/")
def root() -> FileResponse:
    return FileResponse("index.html")

@app.get("/api/registration")
def regisration() -> FileResponse:
    return FileResponse("registration.html")


# Обработчик для создания пользователя
@app.post("/api/registration")
async def create_user(user: User) -> JSONResponse:
    database = Database('server')
    try:
        user = database.add_user(user)
        return JSONResponse(jsonable_encoder(user))
    except Exception as e:
        raise HTTPException(status_code=401, detail=e)


# Обработчик для получения всех пользователей
@app.get("/api/users")
async def get_all_users() -> JSONResponse:
    database = Database('server')
    users = database.get_all_users()
    return JSONResponse(jsonable_encoder(users))

# Обработчик для проверки существования аккаунта
@app.post("/api/login")
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
        projects = database.add_user_block(project)
        return JSONResponse(jsonable_encoder(projects))
    except Exception as e:
        raise HTTPException(status_code=401, detail=e)



if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)