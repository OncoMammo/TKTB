# Menggunakan image Python 3.12-slim sebagai base image
FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    python3-distutils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy req txt
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

# Menentukan port yang akan digunakan oleh aplikasi Flask
EXPOSE 5000

# Perintah untuk menjalankan aplikasi Flask
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]