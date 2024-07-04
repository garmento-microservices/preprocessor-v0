#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
from __future__ import annotations

import glob
import logging
import os
from abc import abstractmethod
from typing import Any, Callable, Dict

import numpy as np
import torch
from detectron2.config import CfgNode, get_cfg  # type: ignore
from detectron2.data.detection_utils import read_image  # type: ignore
from detectron2.engine.defaults import DefaultPredictor  # type: ignore
from detectron2.structures.instances import Instances  # type: ignore
from detectron2.utils.logger import setup_logger  # type: ignore
from PIL import Image

from .densepose import add_densepose_config
from .densepose.structures import (
    DensePoseChartPredictorOutput,
    DensePoseEmbeddingPredictorOutput,
)
from .densepose.vis.extractor import DensePoseOutputsExtractor, DensePoseResultExtractor


LOGGER_NAME = "apply_net"
logger = logging.getLogger(LOGGER_NAME)

colormap = {
    2: [20, 80, 194],
    3: [4, 98, 224],
    4: [8, 110, 221],
    9: [6, 166, 198],
    10: [22, 173, 184],
    15: [145, 191, 116],
    16: [170, 190, 105],
    17: [191, 188, 97],
    18: [216, 187, 87],
    19: [228, 191, 74],
    20: [240, 198, 60],
    21: [252, 205, 47],
    22: [250, 220, 36],
    23: [251, 235, 25],
    24: [248, 251, 14],
}


class InferenceAction:
    @abstractmethod
    def execute_on_outputs(
        self, context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances
    ): ...

    @abstractmethod
    def create_context(self, args: dict[str, str], cfg: CfgNode) -> dict[str, Any]: ...

    @abstractmethod
    def postexecute(self, context: Dict[str, Any]): ...

    def execute(self, args: dict[str, str]):
        logger.info(f"""Loading config from {args["cfg"]}""")
        cfg = self.setup_config(args["cfg"], args["model"])
        logger.info(f"""Loading model from {args["model"]}""")
        predictor = DefaultPredictor(cfg)
        logger.info(f"""Loading data from {args["input"]}""")
        file_list = self._get_input_file_list(args["input"])
        if len(file_list) == 0:
            logger.warning(f"""No input images for {args["input"]}""")
            return
        context = self.create_context(args, cfg)
        for file_name in file_list:
            img = read_image(file_name, format="BGR")  # predictor expects BGR image.
            with torch.no_grad():
                outputs = predictor(img)["instances"]
                self.execute_on_outputs(
                    context, {"file_name": file_name, "image": img}, outputs
                )
        self.postexecute(context)

    def setup_config(
        self,
        config_fpath: str,
        model_fpath: str,
    ):
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(config_fpath)
        cfg.MODEL.WEIGHTS = model_fpath
        cfg.MODEL.DEVICE = "cpu"
        cfg.freeze()
        return cfg

    def _get_input_file_list(self, input_spec: str):
        if os.path.isdir(input_spec):
            file_list = [
                os.path.join(input_spec, fname)
                for fname in os.listdir(input_spec)
                if os.path.isfile(os.path.join(input_spec, fname))
            ]
        elif os.path.isfile(input_spec):
            file_list = [input_spec]
        else:
            file_list = glob.glob(input_spec)
        return file_list


class DumpAction(InferenceAction):
    """
    Dump action that outputs results to a pickle file
    """

    def execute_on_outputs(
        self, context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances
    ):
        image_fpath = entry["file_name"]
        logger.info(f"Processing {image_fpath}")
        result = {"file_name": image_fpath}
        if outputs.has("scores"):
            result["scores"] = outputs.get("scores").cpu()
        if outputs.has("pred_boxes"):
            result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
            if outputs.has("pred_densepose"):
                extractor: (
                    DensePoseResultExtractor
                    | DensePoseOutputsExtractor
                    | Callable[..., Any]
                ) = lambda *_: tuple()
                if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
                    extractor = DensePoseResultExtractor()
                elif isinstance(
                    outputs.pred_densepose, DensePoseEmbeddingPredictorOutput
                ):
                    extractor = DensePoseOutputsExtractor()
                result["pred_densepose"] = extractor(outputs)[0]
        context["results"].append(result)

    def create_context(self, args: dict[str, str], cfg: CfgNode):
        context = {"results": [], "out_fname": args["output"]}
        return context

    def postexecute(self, context: Dict[str, Any]):
        out_fname = context["out_fname"]
        out_dir = os.path.dirname(out_fname)
        if len(out_dir) > 0 and not os.path.exists(out_dir):
            os.makedirs(out_dir)

        context = context["results"][0]
        densepose_content = [
            context["pred_densepose"][0].labels.cpu().tolist(),
            context["pred_densepose"][0].uv.cpu().tolist(),
            context["pred_boxes_XYXY"][0].cpu().tolist(),
        ]
        img = Image.open(context["file_name"])
        img_w, img_h = img.size
        i = np.array(densepose_content[0])
        seg_img = np.zeros((i.shape[0], i.shape[1], 3))

        for y_idx in range(i.shape[0]):
            for x_idx in range(i.shape[1]):
                if i[y_idx][x_idx] in colormap:
                    seg_img[y_idx][x_idx] = colormap[i[y_idx][x_idx]]
                else:
                    seg_img[y_idx][x_idx] = [0, 0, 0]

        box = densepose_content[2]
        box[2] = box[2] - box[0]
        box[3] = box[3] - box[1]
        x, y, w, h = [int(v) for v in box]
        bg = np.zeros((img_h, img_w, 3))
        bg[y : y + h, x : x + w, :] = seg_img
        bg_img = Image.fromarray(np.uint8(bg), "RGB")
        out_fname = out_fname.replace(".json", ".jpg")
        bg_img.save(out_fname)
        logger.info(f"Done saving to {out_fname}")


def create_args(cfg: str, model: str, input_file: str, output_file: str):
    return {
        "cfg": cfg,
        "model": model,
        "input": input_file,
        "output": output_file,
    }


def run_densepose(
    input_file: str,
    output_file: str,
    config_file: str = "configs/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
    model_file: str = "models/densepose/model_final_162be9.pkl",
):
    """
    Run Densepose on input image and produces output image file.
    :param input_file: Input original JPG image
    :param output_file: Output Densepose JPG file
    """
    
    logger = setup_logger(name=LOGGER_NAME)
    logger.setLevel(logging.INFO)
    args = create_args(
        cfg=config_file,
        model=model_file,
        input_file=input_file,
        output_file=output_file,
    )
    action = DumpAction()
    action.execute(args)


def main():
    cfg = "configs/densepose/densepose_rcnn_R_50_FPN_s1x.yaml"
    model = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
    model = "models/densepose/model_final_162be9.pkl"
    input_file = "testdata/origin.jpg"
    output_file = "testdata/poses.json"

    global logger
    logger = setup_logger(name=LOGGER_NAME)
    logger.setLevel(logging.INFO)
    DumpAction().execute(create_args(cfg, model, input_file, output_file))


if __name__ == "__main__":
    main()
