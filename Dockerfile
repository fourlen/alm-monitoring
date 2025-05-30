FROM python:3.11-slim

# Установим необходимые зависимости
RUN pip install --upgrade pip

# Копируем файлы проекта
WORKDIR /app
COPY . /app

# Установим зависимости проекта
RUN pip install -r requirements.txt

# Открываем порт
EXPOSE 8501

# Запускаем Streamlit
CMD ["streamlit", "run", "ethusdc.py", "--server.port=8501", "--server.address=0.0.0.0"]
