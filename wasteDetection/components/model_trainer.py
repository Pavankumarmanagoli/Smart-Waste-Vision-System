import os
import sys
import yaml
import zipfile
import shutil

from wasteDetection.utils.main_utils import read_yaml_file
from wasteDetection.logger import logging
from wasteDetection.exception import AppException
from wasteDetection.entity.config_entity import ModelTrainerConfig
from wasteDetection.entity.artifacts_entity import ModelTrainerArtifact


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig):
        self.model_trainer_config = model_trainer_config

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            # -----------------------------
            # 1) Unzip downloaded dataset
            # -----------------------------
            if not os.path.exists("data.zip"):
                raise FileNotFoundError(
                    "data.zip not found in project root. "
                    "Data ingestion step must download it first."
                )

            logging.info("Unzipping data.zip")
            with zipfile.ZipFile("data.zip", "r") as z:
                z.extractall(".")
            os.remove("data.zip")

            # -----------------------------
            # 2) Read number of classes from data.yaml
            # -----------------------------
            if not os.path.exists("data.yaml"):
                raise FileNotFoundError(
                    "data.yaml not found after unzipping. "
                    "Expected train/, valid/, and data.yaml from dataset zip."
                )

            with open("data.yaml", "r", encoding="utf-8") as stream:
                num_classes = int(yaml.safe_load(stream)["nc"])

            # -----------------------------
            # 3) Create custom YOLOv5 model yaml with correct nc
            # -----------------------------
            model_config_file_name = self.model_trainer_config.weight_name.split(".")[0]
            logging.info(f"Base model config inferred: {model_config_file_name}.yaml")

            base_cfg_path = os.path.join("yolov5", "models", f"{model_config_file_name}.yaml")
            if not os.path.exists(base_cfg_path):
                raise FileNotFoundError(f"YOLO model config not found: {base_cfg_path}")

            config = read_yaml_file(base_cfg_path)
            config["nc"] = num_classes

            custom_cfg_path = os.path.join("yolov5", "models", f"custom_{model_config_file_name}.yaml")
            with open(custom_cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, sort_keys=False)

            # -----------------------------
            # 4) Train YOLOv5 (Windows-safe call)
            # -----------------------------
            # NOTE: We call "python yolov5/train.py ..." directly (no cd, no linux commands)
            train_cmd = (
                f'python yolov5/train.py --img 416 '
                f'--batch {self.model_trainer_config.batch_size} '
                f'--epochs {self.model_trainer_config.no_epochs} '
                f'--data data.yaml '
                f'--cfg "{custom_cfg_path}" '
                f'--weights {self.model_trainer_config.weight_name} '
                f'--name yolov5s_results --cache'
            )
            logging.info(f"Training command: {train_cmd}")
            exit_code = os.system(train_cmd)
            if exit_code != 0:
                raise RuntimeError("YOLOv5 training failed. Check terminal logs above.")

            # -----------------------------
            # 5) Copy best.pt to yolov5/ and artifacts
            # -----------------------------
            src_best = os.path.join(
                "yolov5", "runs", "train", "yolov5s_results", "weights", "best.pt"
            )
            if not os.path.exists(src_best):
                raise FileNotFoundError(f"Trained model not found: {src_best}")

            dst_yolo = os.path.join("yolov5", "best.pt")
            shutil.copyfile(src_best, dst_yolo)

            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            dst_art = os.path.join(self.model_trainer_config.model_trainer_dir, "best.pt")
            shutil.copyfile(src_best, dst_art)

            # -----------------------------
            # 6) Cleanup (Windows-safe)
            # -----------------------------
            for p in [os.path.join("yolov5", "runs"), "train", "valid"]:
                if os.path.exists(p):
                    shutil.rmtree(p, ignore_errors=True)

            if os.path.exists("data.yaml"):
                os.remove("data.yaml")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=dst_yolo
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise AppException(e, sys)
