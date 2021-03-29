#!/usr/bin/python
# -*- coding: utf-8 -*-
# Intial imports

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import requests
from argparse import ArgumentParser
from io import BytesIO
import mlflow
import torch
from torchvision import models
from torchvision import transforms
from captum.attr import visualization as viz
from captum.attr import LayerGradCam, FeatureAblation, LayerActivation, LayerAttribution
import warnings

warnings.filterwarnings("ignore")

torch.manual_seed(123)
np.random.seed(123)


def get_classes():
    idx_to_labels = {
        "0": "background",
        "1": "aeroplane",
        "2": "bicycle",
        "3": "bird",
        "4": "boat",
        "5": "bottle",
        "6": "bus",
        "7": "car",
        "8": "cat",
        "9": "chair",
        "10": "cow",
        "11": "dining table",
        "12": "dog",
        "13": "horse",
        "14": "moterbike",
        "15": "person",
        "16": "potted plant",
        "17": "sheep",
        "18": "sofa",
        "19": "train",
        "20": "tv/monitor",
    }
    return idx_to_labels


def prepare_data(img_url):
    img_url = dict_args["img_url"]
    image_name = img_url.split("/")[-1]
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))
    print("[INFO] {} Sucessfully Downloaded: ".format(image_name))
    (preproc_img, normalized_inp) = preprocessing(img)
    return (preproc_img, normalized_inp)


def preprocessing(image):
    preprocessing_ = transforms.Compose([transforms.Resize(640), transforms.ToTensor()])
    normalize_ = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    preproc_img = preprocessing_(image)
    normalized_inp = normalize_(preproc_img).unsqueeze(0).to(device)
    return (preproc_img, normalized_inp)


def decode_segmap(image, nc=21):
    label_colors = np.array(
        [  # 0=background
            # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
            # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
            # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
            # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
            (0, 0, 0),
            (128, 0, 0),
            (0, 128, 0),
            (128, 128, 0),
            (0, 0, 128),
            (128, 0, 128),
            (0, 128, 128),
            (128, 128, 128),
            (64, 0, 0),
            (192, 0, 0),
            (64, 128, 0),
            (192, 128, 0),
            (64, 0, 128),
            (192, 0, 128),
            (64, 128, 128),
            (192, 128, 128),
            (0, 64, 0),
            (128, 64, 0),
            (0, 192, 0),
            (128, 192, 0),
            (0, 64, 128),
        ]
    )
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def show_segmentation(normalized_inp):
    normalized_inp.requires_grad = True
    out = fcn(normalized_inp)["out"]

    # Find most likely segmentation class for each pixel.

    out_max = torch.argmax(out, dim=1, keepdim=True)

    # Visualize segmentation output using utility method.

    rgb = decode_segmap(out_max.detach().cpu().squeeze().numpy())
    file_path = "data/Segmentation.png"
    plt.title("Segmentation Predictions")
    plt.imsave(file_path, rgb)
    mlflow.log_image(rgb, "Segmentation.png")
    print("[INFO] Segmentation image saved to path {} :".format(file_path))
    return out_max


def agg_segmentation_wrapper(inp):
    # Creates binary matrix with 1 for original argmax class for each pixel
    # and 0 otherwise. Note that this may change when the input is ablated
    # so we use the original argmax predicted above, out_max.
    model_out = fcn(inp)["out"]
    selected_inds = torch.zeros_like(model_out[0:1]).scatter_(1, out_max, 1)
    return (model_out * selected_inds).sum(dim=(2, 3))


def predictions(normalized_inp):
    preds = agg_segmentation_wrapper(normalized_inp)
    pred_label_idx = torch.nonzero(preds, as_tuple=True)[1]
    idx_to_labels = get_classes()
    predictions = [idx_to_labels[str(idx.item())] for idx in pred_label_idx[1:]]
    print("[INFO] Predictions in given image: {}".format(predictions))

    return (predictions, idx_to_labels)


def get_model(pretrained=True):
    fcn = models.segmentation.fcn_resnet101(pretrained=pretrained).to(device).eval()
    return fcn


def attributions_visualization(img_attr, file_name, title):
    (fig, _) = viz.visualize_image_attr(
        img_attr[0].cpu().permute(1, 2, 0).detach().numpy(),
        sign="all",
        title=title,
        use_pyplot=False,
    )
    fig.savefig("data/" + file_name + ".png")
    print("[INFO] {} saved to path {} :".format(file_name + ".png", "data/"))
    mlflow.log_figure(fig, file_name + ".png")


def attributions_visualization_overlay(img_attr, normalized_inp, titles):
    upsampled_img_attr = LayerAttribution.interpolate(img_attr, normalized_inp.shape[2:])
    (fig, _) = viz.visualize_image_attr_multiple(
        upsampled_img_attr[0].cpu().permute(1, 2, 0).detach().numpy(),
        original_image=preproc_img.permute(1, 2, 0).numpy(),
        signs=["all", "positive", "negative"],
        methods=["original_image", "blended_heat_map", "blended_heat_map"],
        titles=titles,
        use_pyplot=False,
    )
    file_path = "data/" + "Overlay_LayerGradCam.png"
    fig.savefig(file_path)
    print("[INFO] Overlayed LayerGradCam image saved to path {} : ".format(file_path))
    mlflow.log_figure(fig, "Overlay_LayerGradCam.png")


def lgc_attributions(normalized_inp):
    lgc = LayerGradCam(agg_segmentation_wrapper, fcn.backbone.layer4[2].conv3)
    gc_attr = lgc.attribute(normalized_inp, target=dict_args["target"])
    la = LayerActivation(agg_segmentation_wrapper, fcn.backbone.layer4[2].conv3)
    activation = la.attribute(normalized_inp)
    return (gc_attr, activation)


def ablate_features(normalized_inp):
    fa = FeatureAblation(agg_segmentation_wrapper)
    fa_attr = fa.attribute(
        normalized_inp, feature_mask=out_max, perturbations_per_eval=2, target=dict_args["target"]
    )
    fa_attr_without_max = (1 - (out_max == dict_args["target"]).float())[0] * fa_attr
    return (fa_attr, fa_attr_without_max)


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    with mlflow.start_run(run_name="Sementic_Segmentation_Captum_mlflow"):
        parser = ArgumentParser(description="Sementic segmentation models's Captum Interpretations")
        parser.add_argument(
            "--img_url", default=True, metavar="string", help="Use pretrained model"
        )
        parser.add_argument(
            "--target", type=int, default=6, metavar="N", help="Class need to be analyse"
        )
        args = parser.parse_args()
        dict_args = vars(args)
        fcn = get_model()
        (preproc_img, normalized_inp) = prepare_data(dict_args["img_url"])
        out_max = show_segmentation(normalized_inp)
        (predictions, idx_to_labels) = predictions(normalized_inp)
        (gc_attr, activation) = lgc_attributions(normalized_inp)
        attributions_visualization_overlay(
            gc_attr,
            normalized_inp,
            titles=["Original Image", "Positive GradCAM ", "Negative GradCAM"],
        )
        (fa_attr, fa_attr_without_max) = ablate_features(normalized_inp)
        attributions_visualization(
            fa_attr,
            file_name="Imp_Regoins_based_on_Feature_Ablation",
            title="Imp Regions for target class: " + idx_to_labels[str(dict_args["target"])],
        )

        attributions_visualization(
            fa_attr_without_max,
            file_name="Relative_Imp_Regions_based_on_Feature_Ablation",
            title="Relative Imp remaining regions for target class: "
            + idx_to_labels[str(dict_args["target"])],
        )
