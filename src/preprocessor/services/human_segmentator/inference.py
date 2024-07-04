import os
import timeit
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms  # type: ignore

from .dataloaders import custom_transforms
from .networks import deeplab_xception_transfer, graph

label_colours = [
    (0, 0, 0),
    (128, 0, 0),
    (255, 0, 0),
    (0, 85, 0),
    (170, 0, 51),
    (255, 85, 0),
    (0, 0, 85),
    (0, 119, 221),
    (85, 85, 0),
    (0, 85, 85),
    (85, 51, 0),
    (52, 86, 128),
    (0, 128, 0),
    (0, 0, 255),
    (51, 170, 221),
    (0, 255, 255),
    (85, 255, 170),
    (170, 255, 85),
    (255, 255, 0),
    (255, 170, 0),
]


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(
        x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device
    )
    return x[tuple(indices)]


def reorder_tail_tensors(tensors: Sequence[torch.Tensor] | torch.Tensor):
    """
    For CIHP
    :param tail_list: tail_list size is 1 x n_class x h x w
    :return:
    """
    # reorder from index 14->19
    reordered = [
        *(tensors[index].unsqueeze(0) for index in range(14)),
        tensors[15].unsqueeze(0),  # 14
        tensors[14].unsqueeze(0),  # 15
        tensors[17].unsqueeze(0),  # 16
        tensors[16].unsqueeze(0),  # 17
        tensors[19].unsqueeze(0),  # 18
        tensors[18].unsqueeze(0),  # 19
    ]

    return torch.cat(reordered, dim=0)


def decode_labels(mask, num_images=1, num_classes=20):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    n, h, w = mask.shape
    assert (
        n >= num_images
    ), "Batch size %d should be greater or equal than number of images to save %d." % (
        n,
        num_images,
    )
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new("RGB", (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs


def read_img(img_path: str):
    return Image.open(img_path).convert("RGB")  # return is RGB pic


def apply_transform(img, transform=lambda *_, **__: None):
    return transform({"image": img, "label": 0})



def to_seg_grayscale(image: Image.Image):
    img_w, img_h = image.size

    img = np.array(image)
    gray_img = np.zeros((img_h, img_w))

    for y_idx in range(img.shape[0]):
        for x_idx in range(img.shape[1]):
            tmp = img[y_idx][x_idx]
            if np.array_equal(tmp, [0, 0, 0]):
                gray_img[y_idx][x_idx] = 0
            if np.array_equal(tmp, [255, 0, 0]):
                gray_img[y_idx][x_idx] = 2  # 머리카락
            elif np.array_equal(tmp, [0, 0, 255]):
                gray_img[y_idx][x_idx] = 13  # 머리
            elif np.array_equal(tmp, [85, 51, 0]):
                gray_img[y_idx][x_idx] = 10  # 목
            elif np.array_equal(tmp, [255, 85, 0]):
                gray_img[y_idx][x_idx] = 5  # 몸통
            elif np.array_equal(tmp, [0, 255, 255]):
                gray_img[y_idx][x_idx] = 15  # 왼팔
            elif np.array_equal(tmp, [51, 170, 221]):
                gray_img[y_idx][x_idx] = 14  # 오른팔
            elif np.array_equal(tmp, [0, 85, 85]):
                gray_img[y_idx][x_idx] = 9  # 바지
            elif np.array_equal(tmp, [0, 0, 85]):
                gray_img[y_idx][x_idx] = 6  # 원피스
            elif np.array_equal(tmp, [0, 128, 0]):
                gray_img[y_idx][x_idx] = 12  # 치마
            elif np.array_equal(tmp, [177, 255, 85]):
                gray_img[y_idx][x_idx] = 17  # 왼다리
            elif np.array_equal(tmp, [85, 255, 170]):
                gray_img[y_idx][x_idx] = 16  # 오른다리
            elif np.array_equal(tmp, [0, 119, 221]):
                gray_img[y_idx][x_idx] = 5  # 외투
            else:
                gray_img[y_idx][x_idx] = 0

    img = cv2.resize(gray_img, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(np.array(img, dtype=np.uint8), "L")


def inference(
    net: torch.nn.Module,
    img_path: str,
    output_path: str,
    output_name: str,
    device="cpu",
    size=(384, 512),
):
    """
    :param net:
    :param img_path:
    :param output_path:
    :return:
    """
    output_name, _ = os.path.splitext(output_name)
    # adj
    adj2_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
    adj2_test = (
        adj2_.unsqueeze(0)
        .unsqueeze(0)
        .expand(1, 1, 7, 20)
        .to(torch.device(device))
        .transpose(2, 3)
    )

    adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
    adj3_test = (
        adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7).to(torch.device(device))
    )

    cihp_adj = graph.preprocess_adj(graph.cihp_graph)
    adj3_ = Variable(torch.from_numpy(cihp_adj).float())
    adj1_test = (
        adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20).to(torch.device(device))
    )

    # multi-scale
    scales: list[float] = [1, 0.5, 0.75, 1.25, 1.5, 1.75]
    img = read_img(img_path)
    img = img.resize(size)
    image_samples = []
    flipped_image_samples = []
    for scale in scales:
        common_transforms = [
            custom_transforms.Scale_only_img(scale),
            custom_transforms.Normalize_xception_tf_only_img(),
            custom_transforms.ToTensor_only_img(),
        ]
        common_transforms_with_flip = [
            custom_transforms.HorizontalFlip_only_img(),
            *common_transforms,
        ]

        composed_transforms_ts = transforms.Compose(common_transforms)
        composed_transforms_ts_flip = transforms.Compose(common_transforms_with_flip)

        image_samples.append(apply_transform(img, composed_transforms_ts))
        flipped_image_samples.append(apply_transform(img, composed_transforms_ts_flip))

    start_time = timeit.default_timer()
    net.eval()

    for idx, sample_batched in enumerate(zip(image_samples, flipped_image_samples)):
        sample = sample_batched[0]["image"].unsqueeze(0)
        sample_flipped = sample_batched[1]["image"].unsqueeze(0)
        inputs = torch.cat((sample, sample_flipped), dim=0)
        if idx == 0:
            _, _, h, w = inputs.size()
        # assert inputs.size() == inputs_f.size()

        # Forward pass of the mini-batch
        inputs = Variable(inputs, requires_grad=False)

        with torch.no_grad():
            # outputs = net.forward(inputs)
            outputs: torch.Tensor = net.forward(
                inputs.to(torch.device(device)),
                adj1_test.to(torch.device(device)),
                adj3_test.to(torch.device(device)),
                adj2_test.to(torch.device(device)),
            )
            outputs = (outputs[0] + flip(reorder_tail_tensors(outputs[1]), dim=-1)) / 2
            outputs = outputs.unsqueeze(0)

            if idx == 0:
                outputs_final = outputs.clone()
            else:
                outputs = F.upsample(
                    outputs, size=(h, w), mode="bilinear", align_corners=True
                )
                outputs_final = outputs_final + outputs
    ################ plot pic
    predictions: torch.Tensor = torch.max(outputs_final, 1)[1]
    results = predictions.cpu().numpy()
    vis_res = decode_labels(results)

    # output_image = Image.fromarray(vis_res[0])
    # output_image = to_seg_grayscale(output_image)
    # output_image.save(f"{output_path}/{output_name}.jpg")
    cv2.imwrite(f"{output_path}/{output_name}.jpg", results[0, :, :])

    end_time = timeit.default_timer()
    print(
        "time used for the multi-scale image inference"
        + " is :"
        + str(end_time - start_time)
    )


def default_net():
    return deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(
        n_classes=20,
        hidden_layers=128,
        source_classes=7,
    )


def do_human_segmentation_inference(
    img_path: str,
    output_file: str,
    model_path: str = "models/graphonomy/inference.pth",
    device: str = "cpu",
):
    """
    Do human segmentation inference.
    :param img_path: Path to original image.
    :param output_file: Path to the output human segmentation RGB image.
    :param model_path: Path to the checkpoint.
    """
    state_dict = torch.load(model_path, map_location=torch.device(device))
    net = default_net()
    net.load_source_model(state_dict)
    net.to(torch.device(device))
    print("loaded model:", model_path)
    output_path = os.path.dirname(output_file)
    output_name = os.path.basename(output_file)

    inference(
        net=net,
        img_path=img_path,
        output_path=output_path,
        output_name=output_name,
        device=device,
    )



def main():
    do_human_segmentation_inference(
        img_path="/Users/binhdh/Coding/FSB-Study/thesis/garmento/preprocessor/testdata/origin.jpg",
        output_file="/Users/binhdh/Coding/FSB-Study/thesis/garmento/preprocessor/testdata/output.jpg",
        model_path="/Users/binhdh/Coding/FSB-Study/thesis/garmento/preprocessor/models/graphonomy/inference.pth",
        device="cpu",
    )
