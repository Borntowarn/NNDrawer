import sqlite3 as sl
from typing import *
from pydantic import BaseModel

class User(BaseModel):
    login: str
    password: str
    api: Optional[str] = None
    username: str
    mail: str
    reg_date: str
    first_name: str
    last_name: str
    graduation: Optional[str] = None
    company: Optional[str] = None

class UserIn(BaseModel):
    login: str
    password: str

class UserOut(BaseModel):
    api: Optional[str] = None
    username: str
    mail: str
    first_name: str
    last_name: str
    graduation: Optional[str] = None
    company: Optional[str] = None

class Database:
    
    
    def __init__(self, database_name) -> None:
        self.db = sl.connect(f'{database_name}.db')
        self.cursor = self.db.cursor()
        self.acc_columns = [
            i[1] 
            for i in self.cursor.execute("PRAGMA table_info(acc)").fetchall()
        ]
        self.block_columns = [
            i[1] 
            for i in self.cursor.execute("PRAGMA table_info(blocks)").fetchall()
        ]
    
    def get_all_users(self):
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
    
    def get_user(self, user: UserIn) -> UserOut:
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

    
    def add_user(self, user: User):
        query = """INSERT INTO acc({}) VALUES({})"""
        cols = ', '.join(user.dict(exclude_defaults=True).keys())
        vals = tuple(user.dict(exclude_defaults=True).values())
        try:
            self.cursor.execute(query.format(cols, ', '.join(['?']*len(vals))), vals)
            self.db.commit()
            return self.get_user(UserIn(login=user.login, password=user.password))
        except sl.OperationalError as e:
            raise e


if __name__ == '__main__':
    db = Database('server')
    