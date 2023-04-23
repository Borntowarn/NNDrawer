from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse, FileResponse
import sqlite3 as sl
import uvicorn
import json

app = FastAPI(debug=True)
cursor = None

def get_user(username):
    cursor.execute(f'SELECT * FROM acc WHERE username = "{username}"')
    return cursor.fetchall()

@app.get("/")
def root():
    global cursor
    db = sl.connect('server.db')
    cursor = db.cursor()
    return FileResponse("index.html")

@app.post("/user/get")
def user(data = Body()):
    result = get_user(data['username'])
    return {"message": result}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)