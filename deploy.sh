#!/usr/bin/env sh

uv sync
uv run alembic upgrade head

npm install
npm run build

uv run fastapi run