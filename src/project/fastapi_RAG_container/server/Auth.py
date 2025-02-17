from pydantic import BaseModel
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import JWTError, jwt
from typing import Optional
from fastapi.security import OAuth2PasswordBearer
from fastapi import Depends, HTTPException, status
from dotenv import load_dotenv
import os

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Jwt Authentication:
# class Token(BaseModel):
#     access_token: str
#     token_type:str

class TokenData(BaseModel):
    username: str | None = None

class User(BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool| None = None
    hashed_pwd: str

# Password Hashing:
Password_Context = CryptContext(schemes = ["bcrypt"])

# Helper Functions:
def Verify_Password(plain_password, hashed_password):
    return Password_Context.verify(plain_password, hashed_password)

def Get_Password_Hash(Password):
    return Password_Context.hash(Password)

# Example DB:
Fake_Users_DB = {
    "testuser": User(username = "testuser", email = "Omar@gmail.com", full_name = "Omar Lotfy", hashed_pwd = Get_Password_Hash("password"))
}

# Authenticate User --> Pwd:
async def Authenticate_User(username: str, password: str):
    User = Fake_Users_DB.get(username)
    if User and Verify_Password(password, User.hashed_pwd):
        return User
    return False

# Creating and Decoding JWTs
async def Create_Access_Token(data: dict, expires_delta: Optional[timedelta] = None):
    To_Encode = data.copy()
    if expires_delta:
        Expire = datetime.utcnow() + expires_delta
    else:
        Expire = datetime.utcnow() + timedelta(minutes = ACCESS_TOKEN_EXPIRE_MINUTES)
    
    To_Encode.update({"exp": Expire})
    return jwt.encode(To_Encode, SECRET_KEY, algorithm = ALGORITHM)


async def Decode_Access_Token(token: str):
    try:
        Payload = jwt.decode(token, SECRET_KEY, algorithms = [ALGORITHM])
        return Payload.get("sub")
    except JWTError:
        return None

# OAuth2 Dependency:
OAuth2_Scheme = OAuth2PasswordBearer(tokenUrl = "token")

async def Get_Current_User(token: str = Depends(OAuth2_Scheme)):
    UserName = Decode_Access_Token(token)
    if UserName is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Invalid Authentication Credentials",
            headers = {"WWW-Authenticate": "Bearer"}
        )
    return UserName
