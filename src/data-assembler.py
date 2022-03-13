import glob
import pandas as pd


class DataAssembler:
    RAW_PATH = "./data/raw/"
    INTERIM_PATH = "./data/interim/"


assembler = DataAssembler()
assembler.assemble()

