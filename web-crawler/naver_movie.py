import json
import sentencepiece as spm
import pandas as pd


def prepare_finetune(vocab, infile, outfile):
    df = pd.read_csv(infile, sep="\t", engine="python")

    with open(outfile, "w") as f:
        for index, row in df.iterrows():
            document = row["document"]
            if type(document) != str:
                continue
            instance = { "id": row["id"], "doc": vocab.encode_as_pieces(document), "label": row["label"] }
            f.write(json.dumps(instance))
            f.write("\n")


vocab_path = "kowiki/kowiki.model"
vocab = spm.SentencePieceProcessor()
vocab.load(vocab_path)

prepare_finetune(vocab, "ratings_train.txt", "ratings_train.json")
prepare_finetune(vocab, "ratings_test.txt", "ratings_test.json")

