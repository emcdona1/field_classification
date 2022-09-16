import os
import sys
from pathlib import Path
import pandas as pd


def main(folder_path: Path):
    file_list = pd.DataFrame(columns=['folder', 'filename'])
    for parent, folder, files in os.walk(folder_path):
        parents = [Path(parent).relative_to(folder_path)] * len(files)
        new_files = pd.DataFrame({'folder': parents, 'filename': files})
        file_list = file_list.append(new_files, ignore_index=True)
    file_list.to_csv(Path(folder_path, 'voucher_list.csv'), index=False)


if __name__ == '__main__':
    assert len(sys.argv) == 2, 'Please provide 1 argument: the folder of images to be indexed.'
    folder = Path(sys.argv[1]).absolute()
    assert folder.exists(), f'Not a valid path: {folder}'

    main(folder)
