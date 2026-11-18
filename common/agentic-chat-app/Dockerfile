ARG STREAMLIT_PORT=8501

FROM python:3.12-slim AS base

ARG STREAMLIT_PORT
ENV STREAMLIT_PORT=${STREAMLIT_PORT} \
    STREAMLIT_SERVER_PORT=${STREAMLIT_PORT}

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV STREAMLIT_SERVER_ENABLE_CORS=true \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_CLIENT_SHOW_ERROR_DETAILS=true

COPY requirements.txt /app/requirements.txt
RUN pip install --timeout 60 --no-cache-dir -r requirements.txt

FROM base AS pycaber

# PYCABER_SPEC is the exact pycaber release preview.lock.json certifies;
# ./demo.py build passes ==<version>. An empty spec fails the build on purpose.
ARG PYCABER_SPEC=""
RUN test -n "$PYCABER_SPEC" && pip install --no-cache-dir "pycaber[auto]${PYCABER_SPEC}"

FROM pycaber AS final

ADD . /app
# The zt-mimic directory seed is the demo's single user source. The app reads a
# login user's tenant and groups from it for display and the X-Caber-Tenant
# header (directory_lookup.py); user_config.yaml holds credentials only.
COPY zt_mimic_seed/directory.json /app/zt_mimic_seed/directory.json

EXPOSE ${STREAMLIT_SERVER_PORT}

HEALTHCHECK CMD curl --fail http://localhost:${STREAMLIT_SERVER_PORT}/_stcore/health

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.enableCORS=true", \
     "--server.enableXsrfProtection=false", \
     "--server.enableWebsocketCompression=false", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
