import base64
import hashlib
import pathlib

from run import run

from fastapi import FastAPI
from pydantic import BaseModel

API_CACHE = ".api"

app = FastAPI()

in_dir = f"{API_CACHE}/in"
out_dir = f"{API_CACHE}/out"
pathlib.Path(in_dir).mkdir(parents=True, exist_ok=True)
pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

def b64_to_img(b64_str: str, path: pathlib.Path):
    """Saves a base64 image string to a file"""
    with open(path, "wb") as file:
        # save image.input_b64 to input_path
        file.write(base64.b64decode(b64_str))

def img_to_b64(path: pathlib.Path):
    """Returns a base64 image string from a file"""
    with open(path, "rb") as file:
        # take the output image and return it as base64
        return base64.b64encode(file.read()).decode("utf-8")

class Image(BaseModel):
    """input_b64: base64 representation of input image"""
    input_b64: str
    """model_path: path to saved model"""
    model_path: str
    """model_type: the model type"""
    model_type: str
    """optimize: optimize the model to half-floats on CUDA?"""
    optimize: bool | None
    """side: RGB and depth side by side in output images?"""
    side: bool | None
    """height: inference encoder image height"""
    height: int | None
    """square: resize to a square resolution?"""
    square: bool | None
    """grayscale: use a grayscale colormap?"""
    grayscale: bool | None


@app.post("/")
async def update_item(
    *,
    image: Image | None = None,
):
    # hash the base64 string to get a unique name
    filename = hashlib.sha256(image.input_b64.encode("utf-8")).hexdigest()
    model_short = image.model_type
    input_path = pathlib.Path(f"{in_dir}/{filename}.png")
    output_path = pathlib.Path(f"{out_dir}/{filename}-{model_short}.png")

    b64_to_img(image.input_b64, input_path)

    # required args
    kwargs = {
        "input_path": in_dir,
        "output_path": out_dir,
        "model_path": image.model_path,
        "model_type": image.model_type,
    }
    # optional args
    if image.optimize:
        kwargs["optimize"] = image.optimize
    if image.side:
        kwargs["side"] = image.side
    if image.height:
        kwargs["height"] = image.height
    if image.square:
        kwargs["square"] = image.square
    if image.grayscale:
        kwargs["grayscale"] = image.grayscale

    # will save output to output_path
    run(**kwargs)

    return img_to_b64(output_path)

