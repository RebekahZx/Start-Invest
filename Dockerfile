# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

FROM base AS build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --prefix=/install .
# Train at build time: deterministic (fixed seed), ~10 s, and keeps the runtime image stateless
RUN PYTHONPATH=/install/lib/python3.12/site-packages PATH=/install/bin:$PATH \
    start-invest-train --out /app/artifacts

FROM base AS runtime
COPY --from=build /install /usr/local
COPY --from=build /app/artifacts /app/artifacts
COPY web /app/web
RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app
ENV START_INVEST_BUNDLE=/app/artifacts/bundle.joblib START_INVEST_WEB=/app/web
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status == 200 else 1)"
CMD ["uvicorn", "start_invest.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--proxy-headers"]
