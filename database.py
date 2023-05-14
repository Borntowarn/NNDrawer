import sqlite3 as sl
from typing import *
from pydantic import BaseModel

class User(BaseModel):
    login: str
    password: str
    username: str
    mail: str 
    reg_date: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    graduation: Optional[str] = None
    company: Optional[str] = None

class UserIn(BaseModel):
    login: str
    password: str

class UserOut(BaseModel):
    id: int
    username: str
    mail: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    graduation: Optional[str] = None
    company: Optional[str] = None


class Block(BaseModel):
    idUser: int
    data: str
    descr: Optional[str] = None

class BlockOut(BaseModel):
    data: str
    descr: Optional[str] = None

class Project(BaseModel):
    idUser: int
    data: str
    descr: Optional[str] = None

class ProjectOut(BaseModel):
    data: str
    descr: Optional[str] = None

class Database:
    
    
    def __init__(self, database_name) -> None:
        self.db = sl.connect(f'{database_name}.db')
        self.cursor = self.db.cursor()
    
    def get_all_users(self) -> list[UserOut]:
        query = 'SELECT {} FROM acc'
        users = self.cursor.execute(
            query.format(
                ', '.join(UserOut.__fields__)
            )
        ).fetchall()
        
        result = []
        for user in users:
            user = dict(
                zip(
                    UserOut.__fields__,
                    user
                )
            )
            result.append(UserOut(**user))
        return result

    
    def get_user(self, username: str) -> UserOut:
        query = 'SELECT {} FROM acc WHERE username = "{}"'
        res = self.cursor.execute(
            query.format(
                ', '.join(UserOut.__fields__),
                username
            )
        ).fetchone()
        if res:
            res = dict(
                zip(
                    UserOut.__fields__,
                    res
                )
            ) 
            return UserOut(**res)
    
    
    def auth_user(self, user: UserIn) -> UserOut:
        query = 'SELECT {} FROM acc WHERE login = "{}" AND password = "{}"'
        res = self.cursor.execute(
            query.format(
                ', '.join(UserOut.__fields__),
                user.login,
                user.password
            )
        ).fetchone()
        if res:
            res = dict(
                zip(
                    UserOut.__fields__,
                    res
                )
            ) 
            return UserOut(**res)

    
    def add_user(self, user: User) -> UserOut:
        query = "INSERT INTO acc({}) VALUES({})"
        cols = ', '.join(user.dict(exclude_defaults=True).keys())
        vals = tuple(user.dict(exclude_defaults=True).values())
        try:
            self.cursor.execute(query.format(cols, ', '.join(['?']*len(vals))), vals)
            self.db.commit()
            return self.auth_user(UserIn(login=user.login, password=user.password))
        except sl.OperationalError as e:
            raise e
    
    
    def get_user_blocks(self, user_id: int) -> list[BlockOut]:
        query = f"SELECT data, descr FROM blocks WHERE idUser = {user_id}"
        blocks = self.cursor.execute(query).fetchall()
        result = []
        for block in blocks:
            block = dict(zip(BlockOut.__fields__, block))
            result.append(BlockOut(**block))
        return result
    
    def add_user_block(self, block: Block) -> list[BlockOut]:
        query = "INSERT INTO blocks(idUser, data, descr) VALUES = (?, ?, ?)"
        try:
            blocks = self.cursor.execute(query, tuple(block.dict().values())).fetchall()
            self.db.commit()
            return self.get_user_blocks(block.idUser)
        except sl.OperationalError as e:
            raise e
    
    
    def get_user_projects(self, user_id: int) -> list[ProjectOut]:
        query = f"SELECT data, descr FROM projects WHERE idUser = {user_id}"
        blocks = self.cursor.execute(query).fetchall()
        result = []
        for block in blocks:
            block = dict(zip(ProjectOut.__fields__, block))
            result.append(ProjectOut(**block))
        return result

    
    def add_user_project(self, project: Project) -> list[BlockOut]:
        query = "INSERT INTO projects(idUser, data, descr) VALUES = (?, ?, ?)"
        try:
            project = self.cursor.execute(query, tuple(project.dict().values())).fetchall()
            self.db.commit()
            return self.get_user_projects(project.idUser)
        except sl.OperationalError as e:
            raise e


if __name__ == '__main__':
    db = Database('server')
    