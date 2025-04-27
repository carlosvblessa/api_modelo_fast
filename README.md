# API de Predição de Iris

## Descrição
Esta é uma API Fast que realiza autenticação e predições usando um modelo de aprendizado de máquina treinado com o conjunto de dados Iris. A API utiliza tokens Bearer para proteger endpoints sensíveis, garantindo segurança nas operações.


---

## Endpoints Disponíveis

### **1. Autenticação (`/login`)**
- **Descrição:** Endpoint para autenticação de usuário.
- **Método:** `POST`
- **Corpo da Requisição:**
  ```json
  {
    "username": "admin",
    "password": "secret"
  }
  ```
- **Respostas:**
  - **200 (Sucesso):**
    ```json
    {
      "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
    }
    ```
    - Use o token retornado no cabeçalho `Authorization` para acessar endpoints protegidos.
  - **401 (Erro):**
    ```json
    {
      "error": "Credenciais inválidas."
    }
    ```

---

### **2. Predição (`/predict`)**
- **Descrição:** Realiza uma predição com base nos dados de entrada fornecidos.
- **Método:** `POST`
- **Segurança:** Protegido por token JWT (envie no cabeçalho `Authorization`).
- **Corpo da Requisição:**
  ```json
  {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }
  ```
- **Respostas:**
  - **200 (Sucesso):**
    ```json
    {
      "predicted_class": 0
    }
    ```
    - `predicted_class`: Classe prevista pelo modelo (0, 1 ou 2).
  - **400 (Erro):**
    ```json
    {
      "error": "Dados inválidos."
    }
    ```
  - **401 (Erro):**
    ```json
    {
      "error": "Token inválido ou ausente."
    }
    ```

---

### **3. Lista de Predições (`/predictions`)**
- **Descrição:** Retorna uma lista de predições realizadas anteriormente.
- **Método:** `GET`
- **Segurança:** Protegido por token JWT (envie no cabeçalho `Authorization`).
- **Parâmetros da Query:**
  - `limit` (opcional): Quantidade de registros a serem retornados (padrão = 10).
    - Exemplo: `/predictions?limit=5`
  - `offset` (opcional): A partir de qual registro começar (padrão = 0).
    - Exemplo: `/predictions?offset=10`
- **Respostas:**
  - **200 (Sucesso):**
    ```json
    [
      {
        "id": 1,
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
        "predicted_class": 0,
        "created_at": "2023-10-01T12:34:56Z"
      },
      {
        "id": 2,
        "sepal_length": 6.2,
        "sepal_width": 3.4,
        "petal_length": 5.4,
        "petal_width": 2.3,
        "predicted_class": 2,
        "created_at": "2023-10-01T12:35:10Z"
      }
    ]
    ```
  - **401 (Erro):**
    ```json
    {
      "error": "Token inválido ou ausente."
    }
    ```

---

## Pré-requisitos
Para rodar esta API localmente, você precisará dos seguintes itens instalados:
- Python 3.x
- Dependências listadas no arquivo `requirements.txt`

---

## Instalação
1. Clone o repositório:
   ```bash
   git clone https://github.com/carlosvblessa/api_modelo_fast.git
   cd api_modelo_fast
   ```

2. Crie e ative o ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Execute a API:
   ```bash
   gunicorn api_modelo_fast:app --reload -k uvicorn.workers.UvicornWorker -b 127.0.0.1:8000
   ```

A API estará disponível em `http://127.0.0.1:8000`.

---

## Uso
- Para testar a API, você pode usar ferramentas como [Postman](https://www.postman.com/) ou [cURL](https://curl.se/) ainda em http://127.0.0.1:8000/docs.

---

## Licença
Este projeto está licenciado sob a [MIT License](LICENSE).


