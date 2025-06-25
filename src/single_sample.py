import os
import sys
from model.network_module import ParametersClassifier
from PIL import Image
from train_config import *
import time

def process_single_image(image_path):
    """Process a single image for error detection"""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return False
    
    model = ParametersClassifier.load_from_checkpoint(
        checkpoint_path=os.environ.get("CHECKPOINT_PATH"),
        num_classes=3,
        gpus=1,
    )
    model.eval()

    print("********* CAXTON single image prediction *********")
    print("Flow rate | Lateral speed | Z offset | Hotend")
    print("**************************************************")

    t1 = time.time()

    try:
        pil_img = Image.open(image_path).convert("RGB")
        x = preprocess(pil_img).unsqueeze(0)
        x = x.to(model.device)
        y_hats = model(x)
        y_hat0, y_hat1, y_hat2, y_hat3 = y_hats

        _, preds0 = torch.max(y_hat0, 1)
        _, preds1 = torch.max(y_hat1, 1)
        _, preds2 = torch.max(y_hat2, 1)
        _, preds3 = torch.max(y_hat3, 1)
        preds = torch.stack((preds0, preds1, preds2, preds3)).squeeze()

        preds_str = str(preds.numpy())
        img_basename = os.path.basename(image_path)
        print("Input:", img_basename, "->", "Prediction:", preds_str)

        t2 = time.time()
        print(f"Completed prediction in {t2 - t1:.2f}s")
        return True
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python single_sample.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    success = process_single_image(image_path)
    sys.exit(0 if success else 1)
