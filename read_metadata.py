import numpy as np
import pandas as pd

from util import wavread, wavwrite


def get_list(filename_meta, filename_eval, filename_test=None):
    fid_meta = open(filename_meta, "r")  # filename, type of event
    fid_eval = open(filename_eval, "r")  # filename, type of scene, filename (Unknown)
    # fid_test = open("meta/evaluation_setup/test.txt", "r")  # only filename

    metas = fid_meta.readlines()
    evals = fid_eval.readlines()
    # test = fid_test.readlines()

    list_meta = []
    list_eval = []
    for meta in metas:
        fn_meta, event = meta.split()
        list_meta.append((fn_meta, event))

    for eval in evals:
        fn_eval, scene, fn_unk = eval.split()
        list_eval.append((fn_eval, scene, fn_unk))

    fid_meta.close()
    fid_eval.close()
    # fid_test.close()

    list_meta = pd.DataFrame(list_meta, columns=["fn_meta", "event"])
    list_eval = pd.DataFrame(list_eval, columns=["fn_eval", "scene", "fn_unk"])

    return list_meta, list_eval


def concat_wav(fn_evals, fn_out):

    total_sig = []
    for fn_eval in fn_evals:
        sig, sr, subtype = wavread(fn_eval)
        # print(fn_eval, sr)
        total_sig.append(sig)

    total_sig = np.concatenate(total_sig)
    wavwrite(fn_out + ".wav", total_sig, sr, subtype)


def read_metadata():

    list_meta, list_eval = get_list(
        "./meta/meta.txt", "./meta/evaluation_setup/evaluate.txt"
    )

    fn_list = [fn_unk.split("_")[0] for fn_unk in list_eval["fn_unk"]]
    fn_list = list(set(fn_list))  # 重複した値の削除

    for fn in fn_list:

        # print(fn)
        mask = list_eval["fn_unk"].str.contains(fn)  # 構成するファイルを抽出
        list_eval["sec"] = [
            int(fn_unk.split("_")[1]) for fn_unk in list_eval["fn_unk"]
        ]  # start [s]

        # 連続か判定する処理
        is_continuous = True
        for sec in list_eval[mask].sort_values("sec")["sec"].diff()[1:]:
            if sec != 30.0:
                is_continuous = False
                break

        if is_continuous:  # 連続であればwavファイル書き出し
            # print(list_eval[mask].sort_values("sec")["sec"])
            concat_wav(list_eval[mask].sort_values("sec")["fn_eval"], fn)


if __name__ == "__main__":
    read_metadata()
