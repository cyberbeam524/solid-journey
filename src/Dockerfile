FROM python:3.8

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx

RUN python -m pip install --upgrade pip

WORKDIR /tmp
COPY webserver/requirements.txt .

ENV PYTHONUNBUFFERED=True \
    PORT=${PORT:-9090} \
    PIP_CACHE_DIR=/.cache

RUN --mount=type=cache,target=$PIP_CACHE_DIR \
    pip install -r requirements.txt

COPY webserver/uwsgi.ini /etc/uwsgi/
COPY webserver/supervisord.conf /etc/supervisor/conf.d/

WORKDIR /app

COPY webserver/*.py /app/
COPY *.py /app/

EXPOSE 9090

CMD ["/usr/local/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
