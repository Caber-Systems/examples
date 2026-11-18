FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Install curl for healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user and group
# Create app directory and mcp-files directory with proper permissions
RUN groupadd --system --gid 1987 service-user && \
    useradd --system --uid 1987 --gid service-user -m service-user && \
    mkdir -p /app && \
    mkdir -p /mcp-files && \
    chown -R service-user:service-user /app /mcp-files

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# PYCABER_SPEC is the exact pycaber release preview.lock.json certifies;
# ./demo.py build passes ==<version>. An empty spec fails the build on purpose.
ARG PYCABER_SPEC=""
RUN test -n "$PYCABER_SPEC" && pip install --no-cache-dir "pycaber[auto]${PYCABER_SPEC}"

# Copy application code
COPY app ./app

# Shared Milvus query path, imported by the `search_semantic` tool as
# `shared.milvus_query`. It lives outside this service because it is the single
# framework-free implementation of "embed a query, search a collection" -- the
# ragchat app's copies are Streamlit-coupled (@st.cache_resource) and cannot be
# imported here. The build context is the repo root, so this path is repo-relative.
COPY shared/milvus_query ./shared/milvus_query

# Per-app LLM usage accounting (byte-identical copy of shared/llm_usage.py).
# app/main.py loads it from /app and registers it as shared.milvus_query's
# embedding usage recorder. The usage dir is created here, owned by
# service-user, so the caber-demo-usage named volume mounted on it copies that
# ownership up when it is first created.
COPY llm_usage.py ./llm_usage.py
# save_file (agentic-tools) resolves the acting user's directory group. The
# agentic-app's stdlib-only lookup module and the zt-mimic seed it reads are
# copied in, never duplicated into app/main.py. app/main.py loads
# /app/directory_lookup.py by path; the seed is at the module's IMAGE_SEED_PATH.
COPY directory_lookup.py ./directory_lookup.py
COPY zt_mimic_seed/directory.json /app/zt_mimic_seed/directory.json
RUN mkdir -p /var/lib/caber-demo/usage && \
    chown -R service-user:service-user /var/lib/caber-demo

# Embedded Milvus Lite (agentic-tools, hipaa-ner). entrypoint.sh starts the
# milvus-lite package's server binary on TCP when MCP_EMBEDDED_MILVUS=true and
# otherwise just runs CMD, so other demos built from this image are unchanged.
# find_spec locates the package without importing it (its __init__ needs
# pkg_resources). The ldd check fails the build, not the demo, if the bundled
# binary is missing a shared library on this base image.
COPY --chmod=0755 entrypoint.sh /usr/local/bin/mcp-entrypoint
RUN lib_dir="$(python -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("milvus_lite").origin).parent / "lib")')" && \
    test -x "$lib_dir/milvus" && \
    ln -s "$lib_dir" /opt/milvus-lite && \
    ! LD_LIBRARY_PATH=/opt/milvus-lite ldd /opt/milvus-lite/milvus | grep "not found" && \
    mkdir -p /var/lib/milvus-lite && \
    chown service-user:service-user /var/lib/milvus-lite

# Switch to non-root user
USER service-user

# Expose MCP HTTP port
EXPOSE 3010

ENTRYPOINT ["/usr/local/bin/mcp-entrypoint"]

# Run the Python MCP server
CMD ["python", "app/main.py"]
