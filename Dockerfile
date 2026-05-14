# MedNeXt - Coronary Artery Segmentation (AMS Izziv 2025)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Delovni direktorij
WORKDIR /workspace

# Časovna cona in okolje
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Ljubljana
ENV PYTHONUNBUFFERED=1

# Sistemske odvisnosti (enake kot ams_s3)
RUN apt-get update && apt-get install -y \
    tzdata \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    vim \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# Python odvisnosti
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# nnU-Net (za baseline primerjavo)
RUN pip install nnunetv2

# Kopiranje kode
COPY . .

# Namestitev MedNeXt iz lokalnega git submodula
RUN pip install --no-cache-dir -e mednext/

# Privzet ukaz
CMD ["python3", "-c", "print('MedNeXt container pripravljen.')"]