FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY scripts ./scripts
ENV PYTHONPATH=/app/src STYLEMATCH_ARTICLES=/app/data/articles.csv
RUN python -m stylematch.twin && python -c "print('twin generated')"
EXPOSE 8000
CMD ["uvicorn", "stylematch.api:app", "--host", "0.0.0.0", "--port", "8000"]
