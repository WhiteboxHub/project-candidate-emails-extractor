from pydantic import BaseModel
from typing import Optional, List

class Contact(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None

class JobPosition(BaseModel):
    title: str
    company: str
    location: Optional[str] = None
