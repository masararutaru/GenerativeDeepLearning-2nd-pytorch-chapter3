# Generative Deep Learning 第2版第3章VAEの実装

PyTorchとDocker,gitを使ってVAEの実装を行う自己学習用のプロジェクト。
Dvid Foster著「生成Deep Learning 第2版」（オライリージャパン発行）で書かれているコードをPytorchに書き直した以下のリポジトリを参考にしている。
https://github.com/terrence-ou/Generative-Deep-Learning-2nd-Edition-PyTorch-JAX
This project is based on (https://github.com/terrence-ou/Generative-Deep-Learning-2nd-Edition-PyTorch-JAX)

## ディレクトリ構成
- `src/`: ソースコード
- `data/`: データセット（Docker実行時に自動ダウンロード）
- `.gitignore`: git管理
- `requirements.txt`: 環境定義
- `Dockerfile`: 環境定義
- `docker-compose.yml`: 起動設定
- `README.md`: 使用手引き

## 環境構築(Setup),普段の使い方 
Dockerがインストールされている前提です。以下のコマンドで環境を構築・起動します。
```bash
# イメージのビルドとコンテナの起動(初回起動時・requirements.txt を書き換えた時・Dockerfileを書き換えた時)
docker compose up -d --build
#作業開始
docker compose up -d
#ちょっと変になった時
docker compose restart
#作業終了
docker compose down
```
## 実行の手引き(コード書くときはここ見る（毎日の起動のみすぐ上のコマンド参照）)

### コンテナ外から直接実行する場合
```bash
docker compose exec app python src/train.py
```
#### コンテナにログインして対話モードに入る場合
```bash
docker compose exec app bash 
python -i src/train.py
``` 
書いたファイルを実行した状態から対話モードでdataの形やモデルについて確認できる
抜けたくなったらcontrol + dを押す(2回押したら対話モードから一気に通常のターミナルまで戻る)
学習中に Ctrl + c を1回押すと、
普通ならそこでプログラムが強制終了して終わるが、-i オプションがついている場合は中断されたその瞬間の状態で対話モード（>>>）に入れる

