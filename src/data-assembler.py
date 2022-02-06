# %%
import glob
import pandas as pd
from pyparsing import col

# %%


class DataAssembler:
    RAW_PATH = "./data/raw/"
    INTERIM_PATH = "./data/interim/"
    DOC_ENDING = "doc.txt"
    TAB_ENDING = "tab.csv"
    FILE_ENDINGS = {DOC_ENDING, TAB_ENDING}

    def split_file_names(self):
        raw_files = glob.glob(f"{self.RAW_PATH}*")
        raw_files = [file.removeprefix(self.RAW_PATH) for file in raw_files]
        rad_ext = {}
        for file in raw_files:
            radical = "_".join(file.split("_")[:-1])  # remove everything after last "_"
            ext = file.removeprefix(f"{radical}_")

            if radical in rad_ext.keys():
                rad_ext[radical] += [ext]
            else:
                rad_ext[radical] = [ext]
        return rad_ext

    @property
    def radicals_to_assemble(self):
        rad_ext = self.split_file_names()
        radicals_to_assemble = []
        for radical, suf in rad_ext.items():
            if set(suf) == self.FILE_ENDINGS:
                radicals_to_assemble.append(radical)
        return radicals_to_assemble

    def assemble(self):
        for radical in self.radicals_to_assemble:
            doc_file = "_".join([radical, self.DOC_ENDING])
            csv_file = "_".join([radical, self.TAB_ENDING])

            with open(self.RAW_PATH + doc_file, "r") as f:
                content = f.read()
            content = content.split("\n")[1:-2]
            content = [row.removeprefix("\t") for row in content]
            colnames = [row.split(" ")[0] for row in content]

            df = pd.read_csv(
                self.RAW_PATH + csv_file,
                names=colnames,
            )
            df.to_csv(
                path_or_buf=f"{self.INTERIM_PATH+radical}.csv",
                index=False,
            )


assembler = DataAssembler()
assembler.assemble()

# %%
