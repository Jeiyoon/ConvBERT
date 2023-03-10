# author: https://paul-hyun.github.io/vocab-with-sentencepiece/
import os, argparse, datetime, re, wget, json
import pandas as pd
import sentencepiece as spm

from csv_to_text import csv_to_text
from generate_vocab import spm_train

SEPARATOR = u"\u241D"


""" wiki file 목록을 읽어들임 """
def list_wiki(dirname):
    filepaths = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        filepath = os.path.join(dirname, filename)

        if os.path.isdir(filepath):
            filepaths.extend(list_wiki(filepath))
        else:
            find = re.findall(r"wiki_[0-9][0-9]", filepath)
            if 0 < len(find):
                filepaths.append(filepath)
    return sorted(filepaths)


""" 여러줄띄기(\n\n...) 한줄띄기로(\n) 변경 """
def trim_text(line):
    data = json.loads(line)
    text = data["text"]

    # https://www.daleseo.com/python-filter/
    value = list(filter(lambda x: len(x) > 0, text.split('\n')))
    data["text"] = "\n".join(value)
    return data


""" csv 파일을 제외한 나머지 파일 삭제 """
def del_garbage(dirname):
    filenames = os.listdir(dirname)
    for filename in filenames:
        filepath = os.path.join(dirname, filename)

        if os.path.isdir(filepath):
            del_garbage(filepath)
            os.rmdir(filepath)
        else:
            if not filename.endswith(".csv"):
                os.remove(filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="kowiki", type=str, required=False,
                        help="위키를 저장할 폴더 입니다.")
    parser.add_argument("--corpus_file", default="kowiki.txt", type=str, required=False,
                        help="corpus 파일 이름")
    parser.add_argument("--vocab_size", default=8000, type=int, required=False,
                        help="vocab 크기")

    args = parser.parse_args()

    """ 1. wiki를 저장할 폴더 생성 """
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    try:
        filename = wget.download("https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-meta-current.xml.bz2", args.output)
    except:
        pass

    """ 2. 시스템 명령어 호출로 코드 실행 """
    os.system(f"python WikiExtractor.py -o {args.output} --json {filename}")

    # text 여러줄 띄기를 한줄 띄기로 합침
    dataset = []

    """ 3. wiki file 목록을 읽어들임 """
    filenames = list_wiki(args.output)
    for filename in filenames:
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    """ 4. 여러줄띄기(\n\n...) 한줄띄기로(\n) 변경 """
                    dataset.append(trim_text(line))

    # 자장파일 결정
    now = datetime.datetime.now().strftime("%Y%m%d")
    output = f"{args.output}/kowiki_{now}.csv"

    # 위키저장 (csv)
    if 0 < len(dataset):
        df = pd.DataFrame(data=dataset)
        df.to_csv(output, sep=SEPARATOR, index=False)

    # 파일삭제
    """ 5. csv 파일을 제외한 나머지 파일 삭제 """
    del_garbage(args.output)

    """ 6. csv 파일을 text 파일로 변환 """
    in_file = "kowiki/kowiki_20230210.csv"
    out_file = "kowiki/kowiki.txt"

    csv_to_text(in_file, out_file)

    """ 7. Vocab 파일 만들기 """
    corpus_file = args.corpus_file # "kowiki.txt"
    prefix = corpus_file.replace(".txt", "") # "kowiki"
    vocab_size = args.vocab_size # 8000
    spm_train(corpus_file, prefix, vocab_size)

    """ 8. 테스트 """
    vocab_file = "kowiki/kowiki.model"
    vocab = spm.SentencePieceProcessor()
    vocab.load(vocab_file)

    lines = [
        "겨울이 되어서 날씨가 무척 추워요.",
        "이번 성탄절은 화이트 크리스마스가 될까요?",
        "겨울에 감기 조심하시고 행복한 연말 되세요."
    ]

    for line in lines:
        pieces = vocab.encode_as_pieces(line)
        ids = vocab.encode_as_ids(line)
        print(line)
        print(pieces)
        print(ids)
        print()
