import os
import logging
import datetime
import jwt
import joblib
import numpy as np
from typing import List, Tuple

from fastapi import FastAPI, Depends, HTTPException, status, Request, Body
from fastapi.responses import HTMLResponse 
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Configurações de JWT
JWT_SECRET = os.getenv("JWT_SECRET", "MEUSEGREDOAQUI")
JWT_ALGORITHM = "HS256"
JWT_EXP_DELTA_SECONDS = 3600

# Configurações de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_modelo_fastapi")

# Configurações do banco de dados
DB_URL = os.getenv("DB_URL", "sqlite:///predictions.db")
engine = create_engine(DB_URL, echo=False)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

# Modelo da tabela de previsões
class PredictionSQL(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    sepal_length = Column(Float, nullable=False)
    sepal_width = Column(Float, nullable=False)
    petal_length = Column(Float, nullable=False)
    petal_width = Column(Float, nullable=False)
    predicted_class = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# Cria as tabelas
Base.metadata.create_all(engine)

# Carrega o modelo
model = joblib.load("modelo_iris_LR.pkl")
logger.info("Modelo carregado com sucesso.")

app = FastAPI(
    title="API de Predição de Iris",
    description="API para autenticação e predição usando um modelo de machine learning com FastAPI",
    version="1.0.0",
    docs_url="/docs",       # Mantém o Swagger UI
    redoc_url="/redoc"      # Para desativar:  redoc_url=None
)
security = HTTPBearer()

# Cache de previsões
predictions_cache: dict[Tuple[float, float, float, float], int] = {}

# Credenciais de teste
TEST_USERNAME = os.getenv("TEST_USERNAME", "admin")
TEST_PASSWORD = os.getenv("TEST_PASSWORD", "secret")


# Schemas Pydantic
class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    token: str

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    predicted_class: int

class PredictionDBResponse(BaseModel):
    id: int
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    predicted_class: int
    created_at: datetime.datetime

class HealthResponse(BaseModel):
    status: str
    db: str

# Funções utilitárias

def create_token(username: str) -> str:
    payload = {
        "username": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(seconds=JWT_EXP_DELTA_SECONDS)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expirado",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido",
            headers={"WWW-Authenticate": "Bearer"}
        )

# Endpoints

@app.post(
    "/login",
    response_model=TokenResponse,
    summary="Endpoint para autenticação de usuário",
    tags=["Autenticação"]
)
def login(
    data: LoginRequest = Body(
        ...,
        example={"username": "admin", "password": "secret"}
    )
):
    if data.username == TEST_USERNAME and data.password == TEST_PASSWORD:
        token = create_token(data.username)
        return {"token": token}
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Credenciais inválidas"
    )

@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Realiza predição do modelo Iris",
    tags=["Predição"]
)
async def predict(
    features: IrisFeatures = Body(
        ...,
        example={
            "petal_length": 1.4,
            "petal_width": 0.2,
            "sepal_length": 5.1,
            "sepal_width": 3.5
        }
    ),
    user: dict = Depends(get_current_user)
):
    key = (features.sepal_length, features.sepal_width, features.petal_length, features.petal_width)
    if key in predictions_cache:
        logger.info("Cache hit para %s", key)
        predicted_class = predictions_cache[key]
    else:
        arr = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]])
        prediction = model.predict(arr)
        predicted_class = int(prediction[0])
        predictions_cache[key] = predicted_class
        logger.info("Cache atualizado para %s", key)

    # Armazena no banco
    db: Session = SessionLocal()
    new_pred = PredictionSQL(
        sepal_length=features.sepal_length,
        sepal_width=features.sepal_width,
        petal_length=features.petal_length,
        petal_width=features.petal_width,
        predicted_class=predicted_class
    )
    db.add(new_pred)
    db.commit()
    db.refresh(new_pred)
    db.close()

    return {"predicted_class": predicted_class}


@app.get(
    "/predictions",
    response_model=List[PredictionDBResponse],
    summary="Lista previsões armazenadas",
    tags=["Predições"],
    responses={
        200: {
            "description": "Lista de previsões",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "created_at": "2025-04-22T18:14:32.227762",
                            "id": 22,
                            "petal_length": 3.6,
                            "petal_width": 1,
                            "predicted_class": 1,
                            "sepal_length": 6,
                            "sepal_width": 2.7
                        }
                    ]
                }
            }
        }
    }
)
def list_predictions(
    limit: int = 10,
    offset: int = 0,
    user: dict = Depends(get_current_user)
):
    db: Session = SessionLocal()
    preds = (
        db.query(PredictionSQL)
        .order_by(PredictionSQL.id.desc())
        .limit(limit)
        .offset(offset)
        .all()
    )
    db.close()
    return preds

@app.get(
    "/",
    summary="Página inicial da API",
    tags=["Raiz"],
    response_class=HTMLResponse,
    responses={
        200: {
            "description": "Página HTML de boas-vindas",
            "content": {
                "text/html": {
                    "example": """
<!DOCTYPE html>
<html lang='pt-BR'>
...
</html>"""
                }
            },
        }
    }
)
def root(request: Request):
    base = str(request.base_url).rstrip("/")
    docs_url = f"{base}/docs"
    redoc_url = f"{base}/redoc"
    year = datetime.datetime.utcnow().year
    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head><meta charset="UTF-8"><title>API Iris Prediction</title></head>
<body>
  <h1>🌼 Bem-vindo à API de Previsão Iris com FastAPI!</h1>
   <h2>🔗 Endpoints Disponíveis:</h2>
  <ul>
    <li><code>GET  /</code>            – Gera esta página HTML</li>
    <li><code>POST /login</code>       – Autenticação e geração de token JWT</li>
    <li><code>POST /predict</code>     – Realizar predição (protegido por token JWT)</li>
    <li><code>GET  /predictions</code> – Listar predições (protegido por token JWT)</li>
    <li><code>GET  /health</code>      – Verificar status da API e do Banco</li>
  </ul>

  <h2>📄 Documentação Interativa:</h2>
    <p>
    <a href="{docs_url}" target="_blank">Acesse o Swagger UI</a><br/>
    <a href="{redoc_url}" target="_blank">Acesse o ReDoc UI</a>
  </p>
  
  <div class="footer">
    &copy; {year} API Iris – Desenvolvido com Python + FastApi
  </div>
</body>
</html>"""
    return HTMLResponse(content=html, status_code=200)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check da API e do banco",
    tags=["Infra"],
    responses={
        200: {
            "description": "Status de saúde da API e do banco",
            "content": {
                "application/json": {
                    "example": {"status": "OK", "db": "up"}
                }
            },
        }
    }
)
def health():
    try:
        conn = engine.connect()
        conn.close()
        db_status = "up"
    except Exception as e:
        logger.error("Health DB error: %s", e)
        db_status = "down"

    return {
        "status": "OK" if db_status == "up" else "FAIL",
        "db": db_status
    }
