import torch
import torchvision.models as models
import requests
from PIL import Image
from io import BytesIO
from torchvision import transforms
import json

# Hàm để tải ảnh từ URL
def load_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Kiểm tra lỗi HTTP
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error loading image: {e}")
        return None

# Hàm lấy các nhãn từ file
def load_imagenet_labels(labels_save_path):
    with open(labels_save_path, 'r') as f:
        labels = json.load(f)
    return labels

# Hàm chính
def run(image_url):
    # Đường dẫn tới file mô hình đã lưu
    model_save_path = "model/resnet18.pth"

    # Khởi tạo mô hình ResNet18
    model = models.resnet18()

    # Tải trọng số của mô hình từ file
    model.load_state_dict(torch.load(model_save_path))

    # Tải tên các nhãn từ file
    labels = load_imagenet_labels('model/labels')

    # Đặt mô hình ở chế độ đánh giá
    model.eval()

    # Tải và tiền xử lý hình ảnh
    input_image = load_image_from_url(image_url)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Tạo batch với kích thước 1

    # Kiểm tra xem có GPU để sử dụng không
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    # Thực hiện dự đoán
    with torch.no_grad():
        output = model(input_batch)

    # Kết quả là một tensor với kích thước [batch_size, num_classes]
    # Ví dụ, để lấy nhãn dự đoán
    _, predicted_class = torch.max(output, 1)
    predicted_label = labels[predicted_class.item()]

    result = {
        "predicted_class_index": predicted_class.item(),
        "predicted_label": predicted_label}

    result_json = json.dumps(result, ensure_ascii=False)

    return result_json



image_url = input("Please enter the image URL: ")
result = run(image_url)
print(result)