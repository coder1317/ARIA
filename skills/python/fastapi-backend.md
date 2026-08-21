---
description: Build production FastAPI backends with auth, DB, and tests
triggers: fastapi, api, backend, rest, endpoint
---

# FastAPI Backend Skill

## Project Structure
```
project/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app, CORS, startup
│   ├── config.py        # Settings via pydantic-settings
│   ├── models/          # SQLAlchemy/Pydantic models
│   ├── routes/          # API routers
│   ├── services/        # Business logic
│   ├── dependencies.py  # Auth, DB session
│   └── utils.py         # Helpers
├── tests/
├── alembic/             # DB migrations
├── requirements.txt
└── README.md
```

## Key Patterns
- Use `pydantic-settings` for config (env vars + .env)
- Dependency injection for DB sessions and auth
- SQLAlchemy 2.0+ async with `AsyncSession`
- JWT auth with `python-jose` + `passlib[bcrypt]`
- Alembic for migrations
- `pytest` + `httpx.AsyncClient` for tests

## Template: Main App
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routes import router

app = FastAPI(title=settings.APP_NAME, version=settings.VERSION)
app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS,
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(router, prefix="/api/v1")
```

## Template: Auth
```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer
from jose import jwt

security = HTTPBearer()

async def get_current_user(token = Depends(security)):
    try:
        payload = jwt.decode(token.credentials, settings.SECRET_KEY, algorithms=["HS256"])
        return payload
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
```
