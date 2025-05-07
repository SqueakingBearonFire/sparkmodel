FROM ubuntu:24.10

# Set a non-interactive frontend (useful for automated builds to avoid interactive prompts)
ENV DEBIAN_FRONTEND=noninteractive

ENV TZ "Asia/Shanghai"

# Update system and install required packages
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    --no-install-recommends

# Create a virtual environment
RUN python3 -m venv /venv
#
# Add the virtual environment to the PATH
ENV PATH="/venv/bin:$PATH"

COPY ./requirements.txt ./

RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

WORKDIR /code

COPY . .