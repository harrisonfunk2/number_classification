import torch
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from model import CNN


device = torch.device('cpu')

def load_model():
    model = CNN().to(device)
    state_dict = torch.load('best_cnn_model.pth', map_location=device,
                            weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

MODEL = load_model()


def pre_process_image(image):
    img = Image.open(image)

    # Convert to grayscale and invert
    img = img.convert('L')
    img = ImageOps.invert(img)

    bounding_box = img.getbbox()
    if bounding_box is None:
        return None
    img = img.crop(bounding_box)

    # Resize, keep proportions
    img.thumbnail((20, 20))

    canvas = Image.new('L', (28, 28), 0)

    # Center the resized digit
    x = (28 - img.width) // 2
    y = (28 - img.height) // 2
    canvas.paste(img, (x, y))

    tensor_img = transforms.ToTensor()(canvas)
    tensor_img = tensor_img.unsqueeze(0)

    return tensor_img


def predict_digit(tensor_img):
    if tensor_img is None:
        return None

    tensor_img = tensor_img.to(device)

    with torch.no_grad():
        logits = MODEL(tensor_img)
        probabilities = torch.softmax(logits, dim=1)

    confidence, predicted_number = torch.max(probabilities, dim=1)

    return {'Confidence': confidence.item(),
            'Predicted Number': predicted_number.item()}
