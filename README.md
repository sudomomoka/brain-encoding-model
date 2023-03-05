# brain-encoding-model

## データの前処理
### NWJC-BERTへの入力データの作成
収集されたDVDセリフ書き下しデータは，MeCabにより形態素解析され，映画ごとにconull形式で保存されている．
また，取得された脳活動データと対応するように1秒ごとにデータが重複されるように入っている.
ファイルから1秒ごとに文を取り出し，それに対応する時制を付与し，NWJC-BERTへの入力データを作成する．

#### 1.all_sentence_extraction.py
一文ごとに書き下し文が形態素解析されているファイルを使用して，1秒ごとに文を取り出す．
一映画一つのファイルではなく，いくつかに分かれているので，ファイルを変えてプログラムを実行する．
##### ＜実行コマンド＞
python all_sentence_extraction.py

#### 2.sentence_count.py
all_sentence_extraction.pyで作成したファイルを脳活動と対応させるため1秒ごとに整列させる．pre-bert1.py同様にrunごとにファイルを変更させてプログラムを実行する．
##### ＜実行コマンド＞
python sentence_count.py

#### 3.sen_tence.py
sentence_count.py で作成したファイルに "ninjalconll"のファイルを参照しながら時制をつける．時制がついているイベントがなかった場合 [その他] のタグをつける．
##### ＜実行コマンド＞
python sen_tence.py

#### 4.run_tag.py
run_tag.pyで作成したファイルに文ごとのIDをつける．最初の20秒は脳活動データとの関係で使用しないため，削っている．
##### ＜実行コマンド＞
python run_tag.py

### 脳活動データの前処理
#### 1.load_data.py
脳活動データは1秒間隔で計測され，時点毎の脳画像のサイズは $96x96x72(= 663552ボクセル)$となっており，映画ごとにmat形式で保存されている．
マスクデータとして，皮質(cortex)・皮質下部(subcortex)についてのデータも付属されている．
このプログラムは，mat形式の脳活動データの必要な部分のみを取り出し，pickle形式で保存する．
生の脳活動データから必要な部分（大脳皮質部分）だけを取り出す．
被験者（subjectCode）ごと，映画（movid），runidを変更しながら実行する．
maskデータは被験者ごとに違うのでその部分の変更も忘れずに．
最初の20秒は不要部分なので削るようになっている．
##### ＜実行コマンド＞
python load_data.py

## 符号化モデル作成
### NWJC-BERTからの特徴量抽出
＜使用プログラム＞
1.new_temp2.py
2.bert_process_functions.py
3.optimizer_Adamw.py

new_temp2.pyでNWJC-BERTから時間的特徴量を抽出する．
また，**bert_process_functions.py**，**optimizer_Adamw.py**は**new_temp2.py**に必要なプログラム．

マルチステップファインチューニングを行う場合は，このプログラムを用いてBERTを学習させ，そのモデルを保存．
再びこのプログラムを用いて，保存したモデルで目的タスクで学習を行う．
モデルの変更は，BiLSTMクラス内の**self.bert_model**,**state_dict**を変更．
**bert-process-functions.py**，**optimizer-Adamw.py**は同ディレクトリ内に置く．
##### ＜実行コマンド＞
python new_temp2.py
