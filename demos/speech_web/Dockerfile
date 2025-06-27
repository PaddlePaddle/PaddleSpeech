# Stage 1: 构建好的paddle-speech镜像
FROM paddle-speech-base:1.0.0

# Set working directory
WORKDIR /app

# Copy the rest of the backend application code
COPY speech_server/ .
# Show what files were copied
RUN ls -la
# Create source directories
RUN mkdir -p source/vpr source/wav source/db source/model

RUN pip install --upgrade pip && \
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

# Expose port
EXPOSE 8010

# Command to run the application
CMD ["python", "main.py", "--port", "8010"] 