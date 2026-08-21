---
description: Docker containerization and deployment
triggers: docker, container, deploy, compose, dockerfile
---

# Docker Deployment Skill

## Dockerfile Template (Python)
```dockerfile
FROM python:3.12-slim AS base
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Docker Compose Template
```yaml
version: "3.8"
services:
  app:
    build: .
    ports: ["8000:8000"]
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/mydb
    depends_on: [db]
    restart: unless-stopped

  db:
    image: postgres:16-alpine
    volumes: [pgdata:/var/lib/postgresql/data]
    environment:
      POSTGRES_DB: mydb
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass

volumes:
  pgdata:
```

## Best Practices
- Multi-stage builds for smaller images
- `.dockerignore` to exclude .git, __pycache__, .venv
- Use `COPY --chown` for file permissions
- Health checks: `HEALTHCHECK CMD curl -f http://localhost:8000/health`
- Never run as root: `USER appuser`
