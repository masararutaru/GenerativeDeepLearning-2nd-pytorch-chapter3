# 1. ベースイメージ（Python 3.10を使う）
FROM python:3.10-slim

# 2. Linuxのパッケージ更新と必要なツールのインストール
# libgl1 などはOpenCVを動かすために必要なOSレベルのライブラリです
RUN apt-get update && apt-get install -y \
    libgl1\
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. コンテナ内の作業ディレクトリを設定
WORKDIR /app

# 4. ローカルの requirements.txt をコンテナにコピー
COPY requirements.txt .

# 5. Pythonライブラリをインストール
RUN pip install --no-cache-dir -r requirements.txt

# 6. コンテナ起動時に維持する設定（開発用）
CMD ["tail", "-f", "/dev/null"]