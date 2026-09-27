import shutil
import urllib.request
import zipfile
from pathlib import Path

import enlighten
import pycolmap
from pycolmap import logging


def run():
    output_path = Path("test")
    image_path = "models/gt/lego_small/default/predictions/train/e000000/"

    database_path = output_path / "database.db"
    sfm_path = output_path / "sfm"

    output_path.mkdir(exist_ok=True)
    logging.set_log_destination(logging.INFO, output_path / "INFO.log.")  # + time

    if database_path.exists():
        database_path.unlink()
    pycolmap.extract_features(database_path, image_path)
    pycolmap.match_exhaustive(database_path)
    num_images = pycolmap.Database(database_path).num_images

    if sfm_path.exists():
        shutil.rmtree(sfm_path)
    sfm_path.mkdir(exist_ok=True)

    with enlighten.Manager() as manager:
        with manager.counter(total=num_images, desc="Images registered:") as pbar:
            pbar.update(0, force=True)
            recs = pycolmap.incremental_mapping(
                database_path,
                image_path,
                sfm_path,
                initial_image_pair_callback=lambda: pbar.update(2),
                next_image_callback=lambda: pbar.update(1),
            )
    for idx, rec in recs.items():
        logging.info(f"#{idx} {rec.summary()}")


if __name__ == "__main__":
    run()
