import pandas as pd
import matplotlib.pyplot as plt
import pathlib

date = "23-08-18 12-35-30"
date = "23-08-18 18-53-17"
date = "23-08-27 21-24-26"
#date = "23-09-03 02-46-00"
mode = "train"
mode = "val"

df = pd.read_csv("log/" + date + "/" + mode + "/metrics.csv", names=['time', 'picture', 'ci', 'exp_area', 'path_length'], header=None)
df['exp_area'] /= (0.8*0.8*0.1)
df['exp_area'] /= 900
plt.plot(df['time'], df['exp_area'], color="blue", label="Exp Area")

#ラベルの追加
plt.xlabel('Training Steps')
plt.ylabel('Explored area')

#表示範囲の指定
#plt.xlim(0, 50000000)
#plt.ylim(0, 1.0)

#凡例の追加
plt.legend()

#指数表記から普通の表記に変換
plt.ticklabel_format(style='plain',axis='x')
plt.ticklabel_format(style='plain',axis='y')

#フォルダがない場合は、作成
p_dir = pathlib.Path("./result/" + mode + "/exp_area_graph")
if not p_dir.exists():
    p_dir.mkdir(parents=True)

#グラフの保存
plt.savefig('./result/' + mode + '/exp_area_graph/' + date + '.png')

#グラフの表示
plt.show()

print("Showing Exp Area graph is completed.")