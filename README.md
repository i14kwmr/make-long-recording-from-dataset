# make-long-recording-from-dataset
[TUT Acoustic scenes 2016, Evaluation dataset](https://zenodo.org/record/165995#.YWVfeBDP0-Q)からメタ情報を使い，元の録音信号（3-5分）を作成する．

## Usage
1. [本リポジトリ](https://github.com/i14kwmr/make-long-recording-from-dataset)をclone
```
$ git clone git@github.com:i14kwmr/make-long-recording-from-dataset.git
```

2. [TUT Acoustic scenes 2016, Evaluation dataset](https://zenodo.org/record/165995#.YWVfeBDP0-Q)をダウンロードする．

3. ダウンロードしたファイルをaudioフォルダにまとめる．

4. read_metadata.pyを実行すると連結された録音信号が作成される．
```
$ python read_metadata.py
```
